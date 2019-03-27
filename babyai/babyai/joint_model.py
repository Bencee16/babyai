import torch.nn as nn
import torch


class JointModel(nn.Module):
    def __init__(self, teacher, student, memory_dim):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.memory_dim = memory_dim

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.memory_dim


    def forward(self, obs_teacher, obs_student, memory_teacher, memory_student, talk=False, instr_embedding=None):
        if talk:
            comm, memory_teacher= self.teacher(obs_teacher, memory_teacher, instr_embedding)
        else:
            comm = torch.zeros(obs_teacher.image.shape[0], self.teacher.message_length, self.teacher.vocab_size)
            if torch.cuda.is_available():
                comm = comm.cuda()

        model_results = self.student(comm, obs_student, memory_student, instr_embedding)
        model_results['memory_teacher'] = memory_teacher

        return model_results, comm

