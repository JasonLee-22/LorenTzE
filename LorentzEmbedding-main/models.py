import math

import torch
from torch import nn

'''class Complex(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, input_dropout_rate=0.2):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.embedding_dim = embedding_dim

        self.E_ct = nn.Embedding(num_entities, embedding_dim)
        self.E_x = nn.Embedding(num_entities, embedding_dim)
        self.E_y = nn.Embedding(num_entities, embedding_dim)
        self.E_z = nn.Embedding(num_entities, embedding_dim) # E x d

        self.R_v1 = nn.Embedding(num_relations, embedding_dim) # R x d
        self.R_v2 = nn.Embedding(num_relations, embedding_dim)
        self.R_v3 = nn.Embedding(num_relations, embedding_dim)

        self.dropout = nn.Dropout(input_dropout_rate)
        self.loss = nn.BCELoss()

    def forward(self, h_idx, r_idx):
        # h_idx Bx1
        # r_idx Bx1
        h_ct = self.E_ct(h_idx).view(-1, self.embedding_dim)
        h_x = self.E_x(h_idx).view(-1, self.embedding_dim)
        h_y = self.E_y(h_idx).view(-1, self.embedding_dim)
        h_z = self.E_z(h_idx).view(-1, self.embedding_dim)
        r_v1 = self.R_v1(h_idx).view(-1, self.embedding_dim)
        r_v2 = self.R_v2(h_idx).view(-1, self.embedding_dim)
        r_v3 = self.R_v3(h_idx).view(-1, self.embedding_dim)
        # f : x -> [0, c]
        t_ct = 1 * h_ct + r_v1 * h_x + r_v2 * h_y + r_v3 * h_z
        t_x =
        t_y =
        t_z =
        # 1 vs. N , N=E
        score_ct = torch.mm(t_ct, self.E_ct.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_x = torch.mm(t_x, self.E_x.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_y = torch.mm(t_y, self.E_y.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_z = torch.mm(t_z, self.E_z.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score = (score_ct + score_x + score_y + score_z) / 4
        score = score.sigmoid()
        return score'''


class Lorentz(nn.Module):

    def __init__(self, embedding_dim, entities_num, relations_num, drop_out, c=1.0):
        super(Lorentz, self).__init__()
        self.embedding_dim = embedding_dim
        self.entities_num = entities_num
        self.relations_num = relations_num

        self.E_ct = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_x = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_y = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_z = nn.Embedding(self.entities_num, self.embedding_dim)

        self.R_vx = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_vy = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_vz = nn.Embedding(self.relations_num, self.embedding_dim)

        self.drop_out = nn.Dropout(drop_out)
        self.c = c
        self.loss = nn.BCELoss()

    def init(self):
        pass

    def forward(self, head, rel):
        h_ct = self.E_ct(head)
        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)

        r_x = self.R_vx(rel)
        r_y = self.R_vy(rel)
        r_z = self.R_vz(rel)

        r_v = r_x * r_x + r_y * r_y + r_z * r_z
        gamma = 1 / (math.sqrt(1 - r_v * r_v / (self.c * self.c)))

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

        score_ct = torch.mm(t_ct, self.E_ct.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_x = torch.mm(t_x, self.E_x.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_y = torch.mm(t_y, self.E_y.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_z = torch.mm(t_z, self.E_z.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score = (score_ct + score_x + score_y + score_z) / 4
        score = score.sigmoid()
        score = self.drop_out(score)

        return score


class Lorentz_sphere(nn.Module):

    def __init__(self, embedding_dim, entities_num, relations_num, drop_out, c=1.0):
        super(Lorentz_sphere, self).__init__()
        self.embedding_dim = embedding_dim
        self.entities_num = entities_num
        self.relations_num = relations_num

        self.E_ct = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_x = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_y = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_z = nn.Embedding(self.entities_num, self.embedding_dim)

        # self.R_r = nn.Embedding(self.relations_num, self.embedding_dim, max_norm=1)
        self.R_r = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_theta = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_phi = nn.Embedding(self.relations_num, self.embedding_dim)
        # self.c_embedding = nn.Embedding(1, self.embedding_dim)
        self.drop_out = nn.Dropout(drop_out)
        self.c = c
        self.loss = nn.BCELoss()
        # self.bias = 0.01

    def init(self):
        pass

    def forward(self, head, rel):
        h_ct = self.E_ct(head)
        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)
        # print('rel_in: ', rel)
        r_r = self.R_r(rel)
        r_theta = torch.tanh(self.R_theta(rel))
        r_phi = torch.tanh(self.R_phi(rel))

        r_x = r_r * torch.sin(r_theta * 2 * math.pi) * torch.cos(r_phi * 2 * math.pi)
        r_y = r_r * torch.sin(r_theta * 2 * math.pi) * torch.sin(r_phi * 2 * math.pi)
        r_z = r_r * torch.cos(r_theta * 2 * math.pi)

        r_v = torch.tanh(r_r)
        gamma = 1.0 / (torch.sqrt(1.0 - r_v * r_v / (self.c * self.c)))
        '''print('r_V: ', r_v)
        print('rv2: ', r_v * r_v)

        print('gamma: ', gamma)'''

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

        # print(self.E_ct.weight.transpose(1, 0).shape)
        # print(h_ct.shape)
        # print('h_ct:    ', h_ct)
        # print('t_ct:    ', t_ct)
        '''sim_ct = self.E_ct.weight.transpose(1, 0).repeat(h_ct.size(0), 1, 1)
        sim_x = self.E_x.weight.transpose(1, 0).repeat(h_x.size(0), 1, 1)
        sim_y = self.E_y.weight.transpose(1, 0).repeat(h_y.size(0), 1, 1)
        sim_z = self.E_z.weight.transpose(1, 0).repeat(h_z.size(0), 1, 1)

        #print(sim_z.shape)
        score_ct = torch.bmm(t_ct, sim_ct).squeeze(1)  # Bxd, Exd -> BxE,
        score_x = torch.bmm(t_x, sim_x).squeeze(1)   # Bxd, Exd -> BxE,
        score_y = torch.bmm(t_y, sim_y).squeeze(1)   # Bxd, Exd -> BxE,
        score_z = torch.bmm(t_z, sim_z).squeeze(1)  # Bxd, Exd -> BxE,
        score = (score_ct + score_x + score_y + score_z) / 4'''
        # print('t_shape: ', t_ct.shape)
        score_ct = torch.mm(t_ct.squeeze(1), self.E_ct.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_x = torch.mm(t_x.squeeze(1), self.E_x.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_y = torch.mm(t_y.squeeze(1), self.E_y.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_z = torch.mm(t_z.squeeze(1), self.E_z.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score = (score_ct + score_x + score_y + score_z) / 4

        # print(score)
        score = torch.sigmoid(score)
        return score
        # return (t_ct, t_x, t_y, t_z)


class Lorentz_changeable(nn.Module):

    def __init__(self, embedding_dim, entities_num, relations_num, drop_out, c=1.0):
        super(Lorentz_changeable, self).__init__()
        self.embedding_dim = embedding_dim
        self.entities_num = entities_num
        self.relations_num = relations_num

        self.E_ct = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_x = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_y = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_z = nn.Embedding(self.entities_num, self.embedding_dim)

        # self.R_r = nn.Embedding(self.relations_num, self.embedding_dim, max_norm=1)
        self.R_r = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_theta = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_phi = nn.Embedding(self.relations_num, self.embedding_dim)
        self.c_embedding = nn.Parameter(torch.FloatTensor(1, embedding_dim))
        nn.init.xavier_normal_(self.c_embedding)
        self.drop_out = nn.Dropout(drop_out)
        # self.c = c
        self.loss = nn.BCELoss()
        # self.bias = 0.01

    def init(self):
        pass

    def forward(self, head, rel):
        h_ct = self.E_ct(head)
        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)
        # print('rel_in: ', rel)
        r_r = self.R_r(rel)
        r_theta = torch.tanh(self.R_theta(rel))
        r_phi = torch.tanh(self.R_phi(rel))

        r_x = r_r * torch.sin(r_theta * 2 * math.pi) * torch.cos(r_phi * 2 * math.pi)
        r_y = r_r * torch.sin(r_theta * 2 * math.pi) * torch.sin(r_phi * 2 * math.pi)
        r_z = r_r * torch.cos(r_theta * 2 * math.pi)

        c_e = self.c_embedding

        r_v = torch.tanh(r_r) * c_e
        gamma = 1.0 / (torch.sqrt(1.0 - r_v * r_v / (c_e * c_e)))

        '''print('r_V: ', r_v)
        print('rv2: ', r_v * r_v)

        print('gamma: ', gamma)'''

        t_ct = gamma * h_ct + gamma * r_x * h_x / c_e + gamma * r_y * h_y / c_e + gamma * r_z * h_z / c_e

        t_x = r_x * gamma * h_ct / c_e + \
              (1 + (r_x * r_x * (gamma - 1) / (r_v * r_v))) * h_x + \
              (r_x * r_y * (gamma - 1) / (r_v * r_v)) * h_y + \
              (r_x * r_z * (gamma - 1) / (r_v * r_v)) * h_z

        t_y = r_y * gamma * h_ct / c_e + \
              (r_x * r_y * (gamma - 1) / (r_v * r_v)) * h_x + \
              (1 + (r_y * r_y * (gamma - 1) / (r_v * r_v))) * h_y + \
              (r_z * r_y * (gamma - 1) / (r_v * r_v)) * h_z

        t_z = r_z * gamma * h_ct / c_e + \
              (r_x * r_z * (gamma - 1) / (r_v * r_v)) * h_x + \
              (r_y * r_z * (gamma - 1) / (r_v * r_v)) * h_y + \
              (1 + (r_z * r_z * (gamma - 1) / (r_v * r_v))) * h_z

        # print(self.E_ct.weight.transpose(1, 0).shape)
        # print(h_ct.shape)
        # print('h_ct:    ', h_ct)
        # print('t_ct:    ', t_ct)
        '''sim_ct = self.E_ct.weight.transpose(1, 0).repeat(h_ct.size(0), 1, 1)
        sim_x = self.E_x.weight.transpose(1, 0).repeat(h_x.size(0), 1, 1)
        sim_y = self.E_y.weight.transpose(1, 0).repeat(h_y.size(0), 1, 1)
        sim_z = self.E_z.weight.transpose(1, 0).repeat(h_z.size(0), 1, 1)

        #print(sim_z.shape)
        score_ct = torch.bmm(t_ct, sim_ct).squeeze(1)  # Bxd, Exd -> BxE,
        score_x = torch.bmm(t_x, sim_x).squeeze(1)   # Bxd, Exd -> BxE,
        score_y = torch.bmm(t_y, sim_y).squeeze(1)   # Bxd, Exd -> BxE,
        score_z = torch.bmm(t_z, sim_z).squeeze(1)  # Bxd, Exd -> BxE,
        score = (score_ct + score_x + score_y + score_z) / 4'''
        # print('t_shape: ', t_ct.shape)
        score_ct = torch.mm(t_ct.squeeze(1), self.E_ct.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_x = torch.mm(t_x.squeeze(1), self.E_x.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_y = torch.mm(t_y.squeeze(1), self.E_y.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_z = torch.mm(t_z.squeeze(1), self.E_z.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score = (score_ct + score_x + score_y + score_z) / 4

        # print(score)
        score = torch.sigmoid(score)
        return score
        # return (t_ct,


class Lorentz_multi_speed(nn.Module):

    def __init__(self, embedding_dim, entities_num, relations_num, drop_out, c=1.0):
        super(Lorentz_multi_speed, self).__init__()
        self.embedding_dim = embedding_dim
        self.entities_num = entities_num
        self.relations_num = relations_num

        self.E_ct = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_x = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_y = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_z = nn.Embedding(self.entities_num, self.embedding_dim)

        # self.R_r = nn.Embedding(self.relations_num, self.embedding_dim, max_norm=1)
        self.R_r = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_theta = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_phi = nn.Embedding(self.relations_num, self.embedding_dim)
        # self.c_embedding = nn.Parameter(torch.FloatTensor(1, embedding_dim))
        self.c_embedding = nn.Embedding(self.relations_num, self.embedding_dim)
        nn.init.xavier_normal_(self.c_embedding.weight.data)
        self.drop_out = nn.Dropout(drop_out)
        # self.c = c
        self.loss = nn.BCELoss()
        # self.bias = 0.01

    def init(self):
        pass

    def forward(self, head, rel):
        h_ct = self.E_ct(head)
        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)
        # print('rel_in: ', rel)
        r_r = self.R_r(rel)
        r_theta = torch.tanh(self.R_theta(rel))
        r_phi = torch.tanh(self.R_phi(rel))

        r_x = r_r * torch.sin(r_theta * 2 * math.pi) * torch.cos(r_phi * 2 * math.pi)
        r_y = r_r * torch.sin(r_theta * 2 * math.pi) * torch.sin(r_phi * 2 * math.pi)
        r_z = r_r * torch.cos(r_theta * 2 * math.pi)

        c_e = self.c_embedding(rel)

        r_v = torch.tanh(r_r) * c_e
        gamma = 1.0 / (torch.sqrt(1.0 - r_v * r_v / (c_e * c_e)))

        '''print('r_V: ', r_v)
        print('rv2: ', r_v * r_v)

        print('gamma: ', gamma)'''

        t_ct = gamma * h_ct + gamma * r_x * h_x / c_e + gamma * r_y * h_y / c_e + gamma * r_z * h_z / c_e

        t_x = r_x * gamma * h_ct / c_e + \
              (1 + (r_x * r_x * (gamma - 1) / (r_v * r_v))) * h_x + \
              (r_x * r_y * (gamma - 1) / (r_v * r_v)) * h_y + \
              (r_x * r_z * (gamma - 1) / (r_v * r_v)) * h_z

        t_y = r_y * gamma * h_ct / c_e + \
              (r_x * r_y * (gamma - 1) / (r_v * r_v)) * h_x + \
              (1 + (r_y * r_y * (gamma - 1) / (r_v * r_v))) * h_y + \
              (r_z * r_y * (gamma - 1) / (r_v * r_v)) * h_z

        t_z = r_z * gamma * h_ct / c_e + \
              (r_x * r_z * (gamma - 1) / (r_v * r_v)) * h_x + \
              (r_y * r_z * (gamma - 1) / (r_v * r_v)) * h_y + \
              (1 + (r_z * r_z * (gamma - 1) / (r_v * r_v))) * h_z

        # print(self.E_ct.weight.transpose(1, 0).shape)
        # print(h_ct.shape)
        # print('h_ct:    ', h_ct)
        # print('t_ct:    ', t_ct)
        '''sim_ct = self.E_ct.weight.transpose(1, 0).repeat(h_ct.size(0), 1, 1)
        sim_x = self.E_x.weight.transpose(1, 0).repeat(h_x.size(0), 1, 1)
        sim_y = self.E_y.weight.transpose(1, 0).repeat(h_y.size(0), 1, 1)
        sim_z = self.E_z.weight.transpose(1, 0).repeat(h_z.size(0), 1, 1)

        #print(sim_z.shape)
        score_ct = torch.bmm(t_ct, sim_ct).squeeze(1)  # Bxd, Exd -> BxE,
        score_x = torch.bmm(t_x, sim_x).squeeze(1)   # Bxd, Exd -> BxE,
        score_y = torch.bmm(t_y, sim_y).squeeze(1)   # Bxd, Exd -> BxE,
        score_z = torch.bmm(t_z, sim_z).squeeze(1)  # Bxd, Exd -> BxE,
        score = (score_ct + score_x + score_y + score_z) / 4'''
        # print('t_shape: ', t_ct.shape)
        score_ct = torch.mm(t_ct.squeeze(1), self.E_ct.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_x = torch.mm(t_x.squeeze(1), self.E_x.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_y = torch.mm(t_y.squeeze(1), self.E_y.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_z = torch.mm(t_z.squeeze(1), self.E_z.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score = (score_ct + score_x + score_y + score_z) / 4

        # print(score)
        score = torch.sigmoid(score)
        return score
