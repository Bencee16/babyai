from babyai.model import ACModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical


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



class Teacher (ACModel):
    def __init__(self, obs_space, action_space, message_length,
                 image_dim=128, memory_dim=128, instr_dim=128, vocab_size=5,
                 use_instr=False, lang_model="gru", use_memory=False, arch="cnn1",
                 aux_info=None, vocabulary = None, max_tau = 0.2):
        super().__init__(obs_space, action_space, image_dim, memory_dim, instr_dim, use_instr, lang_model, use_memory, arch)
        self.message_length = message_length

        self.vocab_size = vocab_size
        self.actor = None
        self.critic = None

        #vocabulary! understand it better
        if vocabulary is not None:
            self.vocab = vocabulary # Vocabulary object, from obss_preprocessor / None
            self.vocab_idx2word = self.vocab.idx2word
            # Add SOS symbol to vocab/get idx
            self.sos_id = self.vocab['<S>']
        else:
            # if Corrector gets to use own vocabulary (standard)
            self.vocab_idx2word = {i: 'w' + str(i) for i in range(self.message_length)}
            self.sos_id = 0

        # self.comm_lstm = nn.LSTM(self.image_dim, self.comm_lstm_hidden_size, batch_first=True)
        self.word_embedding_decoder = nn.Embedding(num_embeddings=self.vocab_size,
                                                   embedding_dim=self.instr_dim)
        self.decoder_rnn = nn.GRU(input_size=self.instr_dim,
                          hidden_size=self.memory_dim,
                          batch_first=True)

        self.hidden2word = nn.Linear(self.memory_dim, self.vocab_size)
        # self.lstm_cell = nn.LSTMCell(self.image_dim, self.comm_lstm_hidden_size)
        # self.embed_token = nn.Parameter(torch.empty((vocab_size, self.image_dim), dtype=torch.float32))

        self.tau_layer = nn.Sequential(nn.Linear(self.memory_dim, 1),
                                       nn.Softplus())
        self.max_tau = max_tau

        # self.random_corrector = False

        self.apply(initialize_parameters)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, obs, memory, instr_embedding=None, tau=1.2):

        memory_rnn_output, memory, extra_predictions = self._get_embed(obs, memory, instr_embedding)

        batch_size = memory_rnn_output.shape[0]

        message = []
        for i in range(self.message_length):

            if i==0:
                decoder_input = torch.tensor([self.sos_id] * batch_size, dtype=torch.long, device=self.device)
                decoder_input_embedded = self.word_embedding_decoder(decoder_input).unsqueeze(1)
                decoder_hidden = memory_rnn_output.unsqueeze(0)


            decoder_out, decoder_hidden = self.decoder_rnn(decoder_input_embedded, decoder_hidden)
            vocab_scores = self.hidden2word(decoder_out)
            vocab_probs = F.softmax(vocab_scores, -1)

            tau = 1.0 / (self.tau_layer(decoder_hidden).squeeze(0) + self.max_tau)
            tau = tau.expand(-1, self.vocab_size).unsqueeze(1)

            if self.training:
                rohc = RelaxedOneHotCategorical(tau, vocab_probs)
                token = rohc.rsample()

                # Straight-through part
                token_hard = torch.zeros_like(token)
                token_hard.scatter_(-1, torch.argmax(token, dim=-1, keepdim=True), 1.0)
                token = (token_hard - token).detach() + token
            else:
                token = torch.zeros_like(vocab_probs, device=self.device)
                token.scatter_(-1, torch.argmax(vocab_probs, dim=-1, keepdim=True), 1.0)

            message.append(token)

            decoder_input_embedded = torch.matmul(token, self.word_embedding_decoder.weight)

        return torch.stack(message, dim=1).squeeze(2), memory
