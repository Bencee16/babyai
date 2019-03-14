from babyai.model import ACModel
import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSoftmax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.softmax = nn.Softmax(dim)
        self.temp = 1

    def forward(self, x):
        u = torch.rand(x.size())
        if torch.cuda.is_available():
            u = u.cuda()
        y = x -torch.log(-torch.log(u + 1e-20) + 1e-20)
        return self.softmax(y/self.temp)


class Teacher (ACModel):
    def __init__(self, obs_space, action_space, message_length,  comm_lstm_hidden_size,
                 image_dim=128, memory_dim=128, instr_dim=128, vocab_size=5,
                 use_instr=False, lang_model="gru", use_memory=False, arch="cnn1",
                 aux_info=None):
        super().__init__(obs_space, action_space, image_dim, memory_dim, instr_dim, use_instr, lang_model, use_memory, arch)
        self.message_length = message_length
        self.comm_lstm_hidden_size = comm_lstm_hidden_size

        self.vocab_size = vocab_size
        self.actor = None
        self.critic = None

        self.comm_lstm = nn.LSTM(self.image_dim, self.comm_lstm_hidden_size, batch_first=True)
        self.hidden2word = nn.Linear(self.comm_lstm_hidden_size, self.vocab_size)
        self.gumbel_softmax = GumbelSoftmax(2)

        self.use_memory = False

    def forward(self, obs, memory, instr_embedding=None):
        embedding, memory, extra_predictions = self._get_embed(obs, memory, instr_embedding)


        lstm_input = torch.transpose(embedding.unsqueeze(2).repeat(1, 1, self.message_length), 1, 2)
        lstm_out, _ = self.comm_lstm(lstm_input)

        out = self.hidden2word(lstm_out)
        word_probs = self.gumbel_softmax(out)

        return word_probs
