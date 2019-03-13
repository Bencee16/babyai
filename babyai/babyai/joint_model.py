import torch.nn as nn
import torch


class JointModel(nn.Module):
    def __init__(self, teacher, student):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.comm_embed = nn.Linear(self.teacher.vocab_size * self.teacher.message_length, self.student.message_dim)

    @property
    def memory_size(self):
        return 2 * self.teacher.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.student.memory_dim

    def forward(self, obs_teacher, obs_student, memory, talk=False, instr_embedding=None):
        if talk:
            comm = self.teacher(obs_teacher, memory, instr_embedding)
        else:
            comm = torch.zeros(obs_teacher.image.shape[0], self.teacher.message_length, self.teacher.vocab_size)
            if torch.cuda.is_available():
                comm = comm.cuda()
        # Todo separating train time and test time, sample in test time

        message_embedding = self.comm_embed(comm.view(comm.shape[0],-1))

        model_results = self.student(message_embedding, obs_student, memory, instr_embedding)

        return model_results, comm

