from babyai.model import ACModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical

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

        # self.comm_lstm = nn.LSTM(self.image_dim, self.comm_lstm_hidden_size, batch_first=True)
        self.hidden2word = nn.Linear(self.comm_lstm_hidden_size, self.vocab_size)
        # self.gumbel_softmax = GumbelSoftmax(2)
        # self.softmax = nn.Softmax(dim=2)
        self.lstm_cell = nn.LSTMCell(self.image_dim, self.comm_lstm_hidden_size)
        self.embed_token = nn.Parameter(torch.empty((vocab_size, self.image_dim), dtype=torch.float32))

        self.use_memory = False

    def forward(self, obs, memory, instr_embedding=None, tau=1.2):
        # embedding, memory, extra_predictions = self._get_embed(obs, memory, instr_embedding)
        #
        # lstm_input = embedding.unsqueeze(0)
        # for i in range(self.message_length):
        #     if i==0:
        #         out, (hidden, cell) = self.comm_lstm(lstm_input)
        #         lstm_out = out
        #     else:
        #         out, (hidden, cell) = self.comm_lstm(lstm_input, (hidden, cell))
        #         lstm_out = torch.cat((lstm_out, out), dim=0)
        #     lstm_input = out.detach()
        #
        #
        # out = self.hidden2word(lstm_out.transpose(0,1))





        #Diana
        embedding, memory, extra_predictions = self._get_embed(obs, memory, instr_embedding)

        message = []
        for i in range(self.message_length):

            if i==0:
                lstm_input = embedding
                (h,c) = self.lstm_cell(lstm_input)
            else:
                lstm_input = torch.matmul(message[-1], self.embed_token)
                (h,c) = self.lstm_cell(lstm_input, (h,c))
            out = F.softmax(self.hidden2word(h), dim=1)


            if self.training:
                rohc = RelaxedOneHotCategorical(tau, out)
                token = rohc.rsample()

                # Straight-through part
                token_hard = torch.zeros_like(token)
                token_hard.scatter_(-1, torch.argmax(token, dim=-1, keepdim=True), 1.0)
                token = (token_hard - token).detach() + token
            else:
                if self.greedy:
                    _, token = torch.max(p, -1)
                else:
                    token = Categorical(p).sample()

            message.append(token)


        #Diana



        # word_probs = self.gumbel_softmax(out)
        # word_probs = self.softmax(out)
        return torch.stack(message, dim=1)
