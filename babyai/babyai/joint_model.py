import torch.nn as nn
import torch


class JointModel(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student

        self.comm_embed = nn.Embedding(self.teacher.vocab_size, self.student.message_dim)
        self.comm_encoder_rnn = nn.GRU(input_size=self.teacher.instr_dim,
                               hidden_size=self.teacher.instr_dim,
                               batch_first=True)

    @property
    def memory_size(self):
        return 2 * self.teacher.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.student.memory_dim

    def forward(self, obs_teacher, obs_student, memory_teacher, memory_student, talk=False, instr_embedding=None):
        if talk:
            comm, memory_teacher= self.teacher(obs_teacher, memory_teacher, instr_embedding)
        else:
            comm = torch.zeros(obs_teacher.image.shape[0], self.teacher.message_length, self.teacher.vocab_size)
            if torch.cuda.is_available():
                comm = comm.cuda()

        message_embedding = torch.matmul(comm, self.comm_embed.weight)

        _, hidden = self.comm_encoder_rnn(message_embedding)
        message_embedding = hidden[-1]

        model_results = self.student(message_embedding, obs_student, memory_student, instr_embedding)
        model_results['memory_teacher'] = memory_teacher


        # embedding = self.teacher(obs_teacher, memory, talk=False, instr_embedding=None)
        # model_results = self.student(message_embedding, obs_student, memory, instr_embedding)



        return model_results, comm

