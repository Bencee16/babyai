from babyai.model import ACModel
import torch
import torch.nn as nn
import gym
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('GRU') != -1:
        for weight in [m.weight_ih_l0, m.weight_hh_l0]:
            weight.data.normal_(0, 1)
            weight.data *= 1 / torch.sqrt(weight.data.pow(2).sum(1, keepdim=True))
        for bias in [m.bias_ih_l0, m.bias_hh_l0]:
            bias.data.fill_(0)
    elif classname.find('Embedding') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))



# Inspired by FiLMedBlock from https://arxiv.org/abs/1709.07871
class ExpertControllerFiLM(nn.Module):
    def __init__(self, in_features, out_features, in_channels, imm_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=imm_channels, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(imm_channels)
        self.conv2 = nn.Conv2d(in_channels=imm_channels, out_channels=out_features, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)

        self.weight = nn.Linear(in_features, out_features)
        self.bias = nn.Linear(in_features, out_features)

        self.apply(initialize_parameters)

    def forward(self, x, y):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        out = x * self.weight(y).unsqueeze(2).unsqueeze(3) + self.bias(y).unsqueeze(2).unsqueeze(3)
        out = self.bn2(out)
        out = F.relu(out)
        return out




class Student(nn.Module):
    def __init__(self, obs_space, action_space, student_obs_type,
                 message_length, vocab_size,
                 image_dim=128, memory_dim=128, instr_dim=128, comm_encoder_dim=128,
                 dropout_rate=0, use_instr=False, lang_model="gru", use_memory=False, arch="cnn1",
                 aux_info=None):
        super().__init__()

        self.obs_space = obs_space
        self.action_space = action_space
        self.student_obs_type = student_obs_type
        self.message_length = message_length
        self.vocab_size = vocab_size
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim
        self.comm_encoder_dim = comm_encoder_dim
        self.dropout_rate = dropout_rate
        self.use_instr = use_instr
        self.lang_model = lang_model
        self.use_memory = use_memory
        self.arch = arch
        self.aux_info = aux_info

        self.policy_input_size = self.comm_encoder_dim + self.memory_dim

        if arch == "cnn1":
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2, 2)),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=image_dim, kernel_size=(2, 2)),
                nn.ReLU()
            )
        elif arch.startswith("expert_filmcnn"):
            if not self.use_instr:
                raise ValueError("FiLM architecture can be used when instructions are enabled")

            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2, 2), stride=2)
            )
            self.film_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        else:
            raise ValueError("Incorrect architecture name: {}".format(arch))

        # Define instruction embedding
        if self.use_instr:
            if self.lang_model in ['gru', 'bigru', 'attgru']:
                self.word_embedding = nn.Embedding(obs_space["instr"], self.instr_dim)
                if self.lang_model in ['gru', 'bigru', 'attgru']:
                    gru_dim = self.instr_dim
                    if self.lang_model in ['bigru', 'attgru']:
                        gru_dim //= 2
                    self.instr_rnn = nn.GRU(
                        self.instr_dim, gru_dim, batch_first=True,
                        bidirectional=(self.lang_model in ['bigru', 'attgru']))
                    self.final_instr_dim = self.instr_dim
                else:
                    kernel_dim = 64
                    kernel_sizes = [3, 4]
                    self.instr_convs = nn.ModuleList([
                        nn.Conv2d(1, kernel_dim, (K, self.instr_dim)) for K in kernel_sizes])
                    self.final_instr_dim = kernel_dim * len(kernel_sizes)

            if self.lang_model == 'attgru':
                self.memory2key = nn.Linear(self.memory_size, self.final_instr_dim)

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_dim, self.memory_dim)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_instr and not "filmcnn" in arch:
            self.embedding_size += self.final_instr_dim

        if arch.startswith("expert_filmcnn"):
            if arch == "expert_filmcnn":
                num_module = 2
            else:
                num_module = int(arch[(arch.rfind('_') + 1):])
            self.controllers = []
            for ni in range(num_module):
                if ni < num_module - 1:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim,
                        out_features=128, in_channels=128, imm_channels=128)
                else:
                    mod = ExpertControllerFiLM(
                        in_features=self.final_instr_dim, out_features=self.image_dim,
                        in_channels=128, imm_channels=128)
                self.controllers.append(mod)
                self.add_module('FiLM_Controler_' + str(ni), mod)

        # Define head for extra info
        if self.aux_info:
            self.extra_heads = None
            self.add_heads()

        self.comm_embed = nn.Embedding(self.vocab_size,
                                       self.comm_encoder_dim)

        self.comm_encoder_rnn = nn.GRU(input_size=self.comm_encoder_dim,
                               hidden_size=self.comm_encoder_dim,
                               batch_first=True)

        self.dropout = nn.Dropout2d(p=self.dropout_rate)
        self.actor = nn.Sequential(
            nn.Linear(self.policy_input_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )
        self.critic = nn.Sequential(
            nn.Linear(self.policy_input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.apply(initialize_parameters)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, comm, obs, memory, instr_embedding=None):

        message_embedding = torch.matmul(comm, self.comm_embed.weight)

        _, hidden = self.comm_encoder_rnn(message_embedding)
        message_encoded = hidden[-1]


        if self.student_obs_type == "vision":
            # Calculating instruction embedding
            if self.use_instr and instr_embedding is None:
                if self.lang_model == 'gru':
                    _, hidden = self.instr_rnn(self.word_embedding(obs.instr))
                    instr_embedding = hidden[-1]

            # Calculating the image imedding
            x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
            if self.arch.startswith("expert_filmcnn"):
                image_embedding = self.image_conv(x)

                # Calculating FiLM_embedding from image and instruction embedding
                for controler in self.controllers:
                    x = controler(image_embedding, instr_embedding)
                FiLM_embedding = F.relu(self.film_pool(x))
            else:
                FiLM_embedding = self.image_conv(x)

            FiLM_embedding = FiLM_embedding.reshape(FiLM_embedding.shape[0], -1)

            # Going through the memory layer
            if self.use_memory:
                hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
                hidden = self.memory_rnn(FiLM_embedding, hidden)
                embedding = hidden[0]
                memory = torch.cat(hidden, dim=1)
            else:
                embedding = x

            if self.use_instr and not "filmcnn" in self.arch:
                embedding = torch.cat((embedding, instr_embedding), dim=1)

            if hasattr(self, 'aux_info') and self.aux_info:
                extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
            else:
                extra_predictions = dict()

        elif self.student_obs_type == "blind":
            embedding = torch.zeros(message_embedding.shape[0], self.semi_memory_size)
            extra_predictions = {}
            if torch.cuda.is_available():
                embedding = embedding.cuda()
        else:
            raise ValueError("Student observation type must be either vision or blind")

        #Policy part
        policy_input = torch.cat((embedding, message_encoded), dim=1)
        policy_input = self.dropout(policy_input)
        x = self.actor(policy_input)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(policy_input)
        value = x.squeeze(1)

        return {'dist': dist, 'value': value, 'memory_student': memory, 'extra_predictions': extra_predictions}

