from babyai.model import ACModel
import torch
import torch.nn as nn
import gym
import torch.nn.functional as F
from torch.distributions.categorical import Categorical



class Student(ACModel):
    def __init__(self, obs_space, student_obs_type, action_space, message_length,
                 image_dim=128, memory_dim=128, instr_dim=128, message_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False, arch="cnn1",
                 aux_info=None):
        super().__init__(obs_space, action_space, image_dim, memory_dim, instr_dim, use_instr, lang_model, use_memory, arch)

        self.student_obs_type = student_obs_type
        self.message_length = message_length
        self.message_dim = message_dim
        self.dropout = nn.Dropout2d(p=0.5)
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size + self.message_dim, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

    def forward(self, message_embedding, obs, memory, instr_embedding=None):

        if self.student_obs_type == "vision":
            embedding, memory, extra_predictions = self._get_embed(obs, memory, instr_embedding)
        elif self.student_obs_type == "blind":
            embedding = torch.zeros(message_embedding.shape[0], self.semi_memory_size)
            extra_predictions = {}
            if torch.cuda.is_available():
                embedding = embedding.cuda()
        else:
            raise ValueError("Student observation type must be either vision or blind")

        full_embedding = torch.cat((embedding, message_embedding), dim=1)
        policy_input = self.dropout(full_embedding)

        x = self.actor(policy_input)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return {'dist': dist, 'value': value, 'memory': memory, 'extra_predictions': extra_predictions}

