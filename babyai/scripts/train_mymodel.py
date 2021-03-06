#!/usr/bin/env python3

"""
Script to train the agent through reinforcment learning.
"""

import os
import logging
import csv
import json
import gym
import time
import datetime
import torch
import numpy as np
import subprocess
import sys
# sys.path.insert(0, os.getcwd())
# sys.path.insert(0, os.getcwd()+'/../gym-minigrid')
import babyai.utils as utils
import babyai.rl
from babyai.arguments import ArgumentParser
from babyai.teacher import Teacher
from babyai.student import Student
from babyai.joint_model import JointModel

from babyai.evaluate import batch_evaluate
from babyai.utils.agent import ModelAgent


# Parse arguments
parser = ArgumentParser()
parser.add_argument("--algo", default='ppo',
                    help="algorithm to use (default: ppo)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--reward-scale", type=float, default=20.,
                    help="Reward scale multiplier")
parser.add_argument("--gae-lambda", type=float, default=0.99,
                    help="lambda coefficient in GAE formula (default: 0.99, 1 means no gae)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--ppo-epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
args = parser.parse_args()

utils.seed(args.seed)

# Generate environments
envs = []
full_obss = []
for i in range(args.procs):
    env = gym.make(args.env)
    full_obs = env.grid.encode()
    env.seed(100 * args.seed + i)
    envs.append(env)
    full_obss.append(full_obs)

# Define model name
prefix = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

model_name_parts = {
    'prefix': prefix,
    'env': args.env,
    'comm_freq': args.comm_freq,
    'message_length': args.message_length,
    'vocab_size': args.vocab_size,
    'teacher_obs': args.teacher_obs,
    'student_obs_type': args.student_obs_type,
    'dropout_rate': args.dropout,
    'class_weights': args.class_weights
    }

default_model_name = "{prefix}_{env}_{comm_freq}_{message_length}_{vocab_size}_{teacher_obs}_{student_obs_type}_{dropout_rate}_{class_weights}".format(**model_name_parts)
if args.pretrained_model:
    default_model_name = args.pretrained_model + '_pretrained_' + default_model_name
args.model = args.model.format(**model_name_parts) if args.model else default_model_name

utils.configure_logging(args.model)
logger = logging.getLogger(__name__)


preprocess_obss_student = utils.ObssPreprocessor(args.model, "egocentric", args.pretrained_model)
preprocess_obss_teacher = utils.ObssPreprocessor(args.model, args.teacher_obs, args.pretrained_model, envs[0].room_size)


#Define actor-critic model
acmodel = utils.load_model(args.model, raise_not_found=False)
if acmodel is None:
    if args.pretrained_model:
        acmodel = utils.load_model(args.pretrained_model, raise_not_found=True)
    else:
        teacher = Teacher(preprocess_obss_teacher.obs_space,
                          args.message_length,
                          args.vocab_size,
                          args.image_dim,
                          args.memory_dim,
                          args.instr_dim,
                          args.comm_decoder_dim,
                          not args.no_instr,
                          args.instr_arch,
                          not args.no_mem,
                          args.arch)
        student = Student(preprocess_obss_student.obs_space,
                          envs[0].action_space,
                          args.student_obs_type,
                          args.message_length,
                          args.vocab_size,
                          args.image_dim,
                          args.memory_dim,
                          args.instr_dim,
                          args.comm_encoder_dim,
                          args.dropout,
                          not args.no_instr,
                          args.instr_arch,
                          not args.no_mem,
                          args.arch)
        acmodel = JointModel(teacher,
                             student,
                             args.memory_dim
                             )


preprocess_obss_student.vocab.save()
utils.save_model(acmodel, args.model)

### Original code ###

if torch.cuda.is_available():
    acmodel.cuda()

# Define actor-critic algo

reshape_reward = lambda _0, _1, reward, _2: args.reward_scale * reward
if args.algo == "ppo":
    algo = babyai.rl.PPOAlgo(envs, acmodel, args.frames_per_proc, args.discount, args.lr, args.beta1, args.beta2,
                             args.gae_lambda,
                             args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                             args.optim_eps, args.clip_eps, args.ppo_epochs, args.batch_size,
                             preprocess_obss_teacher, preprocess_obss_student,
                             reshape_reward, args.class_weights)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))

# When using extra binary information, more tensors (model params) are initialized compared to when we don't use that.
# Thus, there starts to be a difference in the random state. If we want to avoid it, in order to make sure thatFiLM_Controler_1 = {ExpertControllerFiLM} ExpertControllerFiLM(\n  (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n  (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))\n  (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)\n  (weight): Linear(in_features=128, out_features=128, bias=True)\n  (bias): Linear(in_features=128, out_features=128, bias=True)\n)… View
# the results of supervised-loss-coef=0. and extra-binary-info=0 match, we need to reseed here.

utils.seed(args.seed)

# Restore training status

status_path = os.path.join(utils.get_log_dir(args.model), 'status.json')
if os.path.exists(status_path):
    with open(status_path, 'r') as src:
        status = json.load(src)
else:
    status = {'i': 0,
              'num_episodes': 0,
              'num_frames': 0}

# Define logger and Tensorboard writer and CSV writer

header = (["update", "episodes", "frames", "FPS", "duration"]
          + ["return_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["success_rate"]
          + ["num_frames_" + stat for stat in ['mean', 'std', 'min', 'max']]
          + ["entropy", "value", "policy_loss", "value_loss", "loss", "grad_norm"])
if args.tb:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter(utils.get_log_dir(args.model))
csv_path = os.path.join(utils.get_log_dir(args.model), 'log.csv')
first_created = not os.path.exists(csv_path)
# we don't buffer data going in the csv log, cause we assume
# that one update will take much longer that one write to the log
csv_writer = csv.writer(open(csv_path, 'a', 1))
if first_created:
    csv_writer.writerow(header)

# Log code state, command, availability of CUDA and model

babyai_code = list(babyai.__path__)[0]
try:
    last_commit = subprocess.check_output(
        'cd {}; git log -n1'.format(babyai_code), shell=True).decode('utf-8')
    logger.info('LAST COMMIT INFO:')
    logger.info(last_commit)
except subprocess.CalledProcessError:
    logger.info('Could not figure out the last commit')
try:
    diff = subprocess.check_output(
        'cd {}; git diff'.format(babyai_code), shell=True).decode('utf-8')
    if diff:
        logger.info('GIT DIFF:')
        logger.info(diff)
except subprocess.CalledProcessError:
    logger.info('Could not figure out the last commit')
logger.info('COMMAND LINE ARGS:')
logger.info(args)
logger.info("CUDA available: {}".format(torch.cuda.is_available()))
logger.info(acmodel)

# Train model

total_start_time = time.time()
best_success_rate = 0
test_env_name = args.env
while status['num_frames'] < args.frames:
    # Update parameters
    update_start_time = time.time()
    logs, comms, actions = algo.update_parameters(args.teacher_obs, args.comm_freq)
    update_end_time = time.time()

    status['num_frames'] += logs["num_frames"]
    status['num_episodes'] += logs['episodes_done']
    status['i'] += 1

    #Saving communication and actions
    if args.stats_save_interval != 0 and status['i'] % args.stats_save_interval == 0:
        if not os.path.exists('communication/'+args.model):
            os.makedirs('communication/'+args.model)
        torch.save(comms, './communication/'+args.model+'/'+str(status['i'])+'.pt')

        if not os.path.exists('action/'+args.model):
            os.makedirs('action/'+args.model)
        torch.save(actions, './action/'+args.model+'/'+str(status['i'])+'.pt')

    # Print logs
    if status['i'] % args.log_interval == 0:
        total_ellapsed_time = int(time.time() - total_start_time)
        fps = logs["num_frames"] / (update_end_time - update_start_time)
        duration = datetime.timedelta(seconds=total_ellapsed_time)
        return_per_episode = utils.synthesize(logs["return_per_episode"])
        success_per_episode = utils.synthesize(
            [1 if r > 0 else 0 for r in logs["return_per_episode"]])
        num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

        data = [status['i'], status['num_episodes'], status['num_frames'],
                fps, total_ellapsed_time,
                *return_per_episode.values(),
                success_per_episode['mean'],
                *num_frames_per_episode.values(),
                logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"],
                logs["loss"], logs["grad_norm"]]

        format_str = ("U {} | E {} | F {:06} | FPS {:04.0f} | D {} | R:xsmM {: .2f} {: .2f} {: .2f} {: .2f} | "
                      "S {:.2f} | F:xsmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | "
                      "pL {: .3f} | vL {:.3f} | L {:.3f} | gN {:.3f} | ")

        logger.info(format_str.format(*data))
        if args.tb:
            assert len(header) == len(data)
            for key, value in zip(header, data):
                writer.add_scalar(key, float(value), status['num_frames'])

        csv_writer.writerow(data)

    # Save obss preprocessor vocabulary and model

    if args.save_interval > 0 and status['i'] % args.save_interval == 0:
        print('saving')
        preprocess_obss_student.vocab.save()
        with open(status_path, 'w') as dst:
            json.dump(status, dst)
            utils.save_model(acmodel, args.model)

        # Testing the model before saving
        # agent = ModelAgent(args.model, obss_preprocessor, argmax=True)
        # agent.model = acmodel
        # agent.model.eval()
        # logs = batch_evaluate(agent, test_env_name, args.val_seed, args.val_episodes)
        # agent.model.train()
        # mean_return = np.mean(logs["return_per_episode"])
        # success_rate = np.mean([1 if r > 0 else 0 for r in logs['return_per_episode']])
        # if success_rate > best_success_rate:
        #     best_success_rate = success_rate
        #     utils.save_model(acmodel, args.model + '_best')
        #     obss_preprocessor.vocab.save(utils.get_vocab_path(args.model + '_best'))
        #     logger.info("Return {: .2f}; best model is saved".format(mean_return))
        # else:
        #     logger.info("Return {: .2f}; not the best model; not saved".format(mean_return))