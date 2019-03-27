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






class Teacher (nn.Module):
    def __init__(self, obs_space,
                 message_length,  vocab_size,
                 image_dim=128, memory_dim=128, instr_dim=128, comm_decoder_dim=128,
                 use_instr=False, lang_model="gru", use_memory=False, arch="cnn1",
                 aux_info=None, vocabulary=None, max_tau = 0.2):
        super().__init__()

        self.obs_space = obs_space
        self.message_length = message_length
        self.vocab_size = vocab_size
        self.image_dim = image_dim
        self.memory_dim = memory_dim
        self.instr_dim = instr_dim
        self.comm_decoder_dim = comm_decoder_dim
        self.use_instr = use_instr
        self.lang_model = lang_model
        self.use_memory = use_memory
        self.arch = arch
        self.aux_info = aux_info
        self.max_tau = max_tau


        if vocabulary is not None:
            self.vocab = vocabulary # Vocabulary object, from obss_preprocessor / None
            self.vocab_idx2word = self.vocab.idx2word
            # Add SOS symbol to vocab/get idx
            self.sos_id = self.vocab['<S>']
        else:
            # if Corrector gets to use own vocabulary (standard)
            self.vocab_idx2word = {i: 'w' + str(i) for i in range(self.vocab_size)}
            self.sos_id = 0


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
                if ni < num_module-1:
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


        self.word_embedding_decoder = nn.Embedding(num_embeddings=self.vocab_size,
                                                   embedding_dim=self.comm_decoder_dim)
        self.decoder_rnn = nn.GRU(input_size=self.comm_decoder_dim,
                          hidden_size=self.comm_decoder_dim,
                          batch_first=True)

        self.hidden2word = nn.Linear(self.comm_decoder_dim, self.vocab_size)

        self.tau_layer = nn.Sequential(nn.Linear(self.comm_decoder_dim, 1),
                                       nn.Softplus())

        self.apply(initialize_parameters)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim

    def forward(self, obs, memory, instr_embedding=None, tau=1.2):

        # Calculating instruction embedding
        if self.use_instr and instr_embedding is None:
            if self.lang_model == 'gru':
                _, hidden = self.instr_rnn(self.word_embedding(obs.instr))
                instr_embedding =  hidden[-1]

        #Calculating the image imedding
        x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
        if self.arch.startswith("expert_filmcnn"):
            image_embedding = self.image_conv(x)

            #Calculating FiLM_embedding from image and instruction embedding
            for controler in self.controllers:
                x = controler(image_embedding, instr_embedding)
            FiLM_embedding = F.relu(self.film_pool(x))
        else:
            FiLM_embedding = self.image_conv(x)

        FiLM_embedding = FiLM_embedding.reshape(FiLM_embedding.shape[0], -1)

        #Going through the memory layer
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

        memory_rnn_output = embedding
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
        comm = torch.stack(message, dim=1).squeeze(2)
        return comm, memory



    # def _get_instr_embedding(self, instr):
    #     if self.lang_model == 'gru':
    #         _, hidden = self.instr_rnn(self.word_embedding(instr))
    #         return hidden[-1]
    #
    #
    # def _get_embed(self, obs, memory, instr_embedding):
    #
    #     if self.use_instr and instr_embedding is None:
    #         instr_embedding = self._get_instr_embedding(obs.instr)
    #
    #     x = torch.transpose(torch.transpose(obs.image, 1, 3), 2, 3)
    #     if self.arch.startswith("expert_filmcnn"):
    #         x = self.image_conv(x)
    #         for controler in self.controllers:
    #             x = controler(x, instr_embedding)
    #         x = F.relu(self.film_pool(x))
    #     else:
    #         x = self.image_conv(x)
    #
    #     x = x.reshape(x.shape[0], -1)
    #
    #     if self.use_memory:
    #         hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
    #         hidden = self.memory_rnn(x, hidden)
    #         embedding = hidden[0]
    #         memory = torch.cat(hidden, dim=1)
    #     else:
    #         embedding = x
    #
    #     if self.use_instr and not "filmcnn" in self.arch:
    #         embedding = torch.cat((embedding, instr_embedding), dim=1)
    #
    #     if hasattr(self, 'aux_info') and self.aux_info:
    #         extra_predictions = {info: self.extra_heads[info](embedding) for info in self.extra_heads}
    #     else:
    #         extra_predictions = dict()
    #
    #     return embedding, memory, extra_predictions