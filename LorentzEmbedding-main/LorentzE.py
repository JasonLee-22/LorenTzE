"""
@author: lxy
@email: linxy59@mail2.sysu.edu.cn
@date: 2022/4/12
@description: null
"""
import math

import torch
from torch import nn


class LorentzE(nn.Module):
    def __init__(self, embedding_dim, entities_num, relations_num, drop_out, c=1.0):
        super(LorentzE, self).__init__()
        self.embedding_dim = embedding_dim
        self.entities_num = entities_num
        self.relations_num = relations_num

        self.E_ct = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_x = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_y = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_z = nn.Embedding(self.entities_num, self.embedding_dim)

        self.R_r = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_theta = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_phi = nn.Embedding(self.relations_num, self.embedding_dim)
        # self.c_embedding = nn.Embedding(1, self.embedding_dim)
        self.drop_out = nn.Dropout(drop_out)
        self.c = c
        self.loss = nn.BCELoss()
        # self.bias = 0.01

    def init(self):
        nn.init.xavier_normal_(self.E_c.weight.data)
        nn.init.xavier_normal_(self.E_x.weight.data)
        nn.init.xavier_normal_(self.E_y.weight.data)
        nn.init.xavier_normal_(self.E_z.weight.data)
        nn.init.xavier_normal_(self.R_r.weight.data)
        nn.init.xavier_normal_(self.R_theta.weight.data)
        nn.init.xavier_normal_(self.R_phi.weight.data)

    def forward(self, head, rel):
        h_ct = self.E_ct(head)
        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)

        r_r = self.R_r(rel)
        r_theta = torch.tanh(self.R_theta(rel))
        r_phi = torch.tanh(self.R_phi(rel))

        r_x = r_r * torch.sin(r_theta * 2 * math.pi) * torch.cos(r_phi * 2 * math.pi)
        r_y = r_r * torch.sin(r_theta * 2 * math.pi) * torch.sin(r_phi * 2 * math.pi)
        r_z = r_r * torch.cos(r_theta * 2 * math.pi)

        r_v = torch.tanh(r_r)
        gamma = 1.0 / (torch.sqrt(1.0 - r_v * r_v / (self.c * self.c)))

        t_ct = gamma * h_ct + gamma * r_x * h_x / self.c + gamma * r_y * h_y / self.c + gamma * r_z * h_z / self.c

        t_x = r_x * gamma * h_ct / self.c + \
              (1 + (r_x * r_x * (gamma - 1) / (r_v * r_v))) * h_x + \
              (r_x * r_y * (gamma - 1) / (r_v * r_v)) * h_y + \
              (r_x * r_z * (gamma - 1) / (r_v * r_v)) * h_z

        t_y = r_y * gamma * h_ct / self.c + \
              (r_x * r_y * (gamma - 1) / (r_v * r_v)) * h_x + \
              (1 + (r_y * r_y * (gamma - 1) / (r_v * r_v))) * h_y + \
              (r_z * r_y * (gamma - 1) / (r_v * r_v)) * h_z

        t_z = r_z * gamma * h_ct / self.c + \
              (r_x * r_z * (gamma - 1) / (r_v * r_v)) * h_x + \
              (r_y * r_z * (gamma - 1) / (r_v * r_v)) * h_y + \
              (1 + (r_z * r_z * (gamma - 1) / (r_v * r_v))) * h_z

        print('h_ct:    ', h_ct.shape)
        print('t_ct:    ', t_ct.shape)

        score_ct = torch.mm(t_ct.squeeze(1), self.E_ct.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_x = torch.mm(t_x.squeeze(1), self.E_x.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_y = torch.mm(t_y.squeeze(1), self.E_y.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_z = torch.mm(t_z.squeeze(1), self.E_z.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score = (score_ct + score_x + score_y + score_z) / 4
        score = torch.sigmoid(score)
        return score
