import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math


class LorenTzE(nn.Module):
    def __init__(self, embedding_dim, entities_num, relations_num, timestamps_num, drop_out, c=1.0, alpha = 0.5, bias=-1):
        super(LorenTzE, self).__init__()
        self.embedding_dim = embedding_dim
        self.entities_num = entities_num
        self.relations_num = relations_num
        self.timestamps_num = timestamps_num


        self.T_ct = nn.Embedding(self.timestamps_num, self.embedding_dim)

        self.E_x = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_y = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_z = nn.Embedding(self.entities_num, self.embedding_dim)

        # self.R_r = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_x = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_y = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_z = nn.Embedding(self.relations_num, self.embedding_dim)

        self.E_bn = nn.BatchNorm1d(self.embedding_dim)

        self.b_x = nn.Parameter(torch.zeros(entities_num))
        self.b_y = nn.Parameter(torch.zeros(entities_num))
        self.b_z = nn.Parameter(torch.zeros(entities_num))

        # self.c_embedding = nn.Embedding(1, self.embedding_dim)
        self.drop_out = nn.Dropout(drop_out)
        self.c = c
        self.alpha = alpha
        self.bias = bias
        self.bce = nn.BCELoss()

    def init(self):
        nn.init.xavier_normal_(self.T_ct.weight.data)
        nn.init.xavier_normal_(self.E_x.weight.data)
        nn.init.xavier_normal_(self.E_y.weight.data)
        nn.init.xavier_normal_(self.E_z.weight.data)
        nn.init.xavier_normal_(self.R_x.weight.data)
        nn.init.xavier_normal_(self.R_y.weight.data)
        nn.init.xavier_normal_(self.R_z.weight.data)

    def forward(self, head, rel, timestamp):
        '''head = head.view(-1)
        rel = rel.view(-1)
        timestamp = timestamp.view(-1)'''


        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)

        h_ct = self.T_ct(timestamp)

        h_ct = self.E_bn(h_ct)
        h_x = self.E_bn(h_x)
        h_y = self.E_bn(h_y)
        h_z = self.E_bn(h_z)

        r_x = self.R_x(rel)
        r_y = self.R_y(rel)
        r_z = self.R_z(rel)
        length = torch.sqrt(r_x ** 2 + r_y ** 2 + r_z ** 2).detach()

        h_ct = torch.sin(h_ct)

        r_v_rate = torch.sigmoid(length)

        r_x = r_x / length
        r_y = r_y / length
        r_z = r_z / length

        r_v_rate_2 = r_v_rate * r_v_rate
        gamma = 1.0 / (torch.sqrt(1.0 - r_v_rate_2))

        t_ct = gamma * h_ct + gamma * r_x * r_v_rate * h_x + gamma * r_y * r_v_rate * h_y + gamma * r_z * r_v_rate * h_z

        t_x = r_x * r_v_rate * gamma * h_ct + \
              (1 + (r_x * r_x * (gamma - 1))) * h_x + \
              (r_x * r_y * (gamma - 1)) * h_y + \
              (r_x * r_z * (gamma - 1)) * h_z

        t_y = r_y * r_v_rate * gamma * h_ct + \
              (r_x * r_y * (gamma - 1)) * h_x + \
              (1 + (r_y * r_y * (gamma - 1))) * h_y + \
              (r_z * r_y * (gamma - 1)) * h_z

        t_z = r_z * r_v_rate * gamma * h_ct + \
              (r_x * r_z * (gamma - 1)) * h_x + \
              (r_y * r_z * (gamma - 1)) * h_y + \
              (1 + (r_z * r_z * (gamma - 1))) * h_z

        t_ct = self.drop_out(t_ct)
        t_x = self.drop_out(t_x)
        t_y = self.drop_out(t_y)
        t_z = self.drop_out(t_z)

        '''score_x = torch.sigmoid(torch.mm(t_x, self.E_x.weight.transpose(1, 0)))
        score_y = torch.sigmoid(torch.mm(t_y, self.E_y.weight.transpose(1, 0)))
        score_z = torch.sigmoid(torch.mm(t_z, self.E_z.weight.transpose(1, 0)))'''
        #print("before mm:{}".format(torch.cuda.memory_allocated(0)))
        score_x = torch.mm(t_x, self.E_x.weight.transpose(1, 0))
        score_y = torch.mm(t_y, self.E_y.weight.transpose(1, 0))
        score_z = torch.mm(t_z, self.E_z.weight.transpose(1, 0))
        score_x = score_x + self.b_x.expand_as(score_x)
        score_y = score_y + self.b_y.expand_as(score_y)
        score_z = score_z + self.b_z.expand_as(score_z)
        #print("after mm:{}".format(torch.cuda.memory_allocated(0)))
        '''t_score = torch.sum((h_ct - t_ct)**2, dim=1)
        t_score = F.relu(t_score + self.bias)'''

        return score_x, score_y, score_z

    def loss(self, scores, target):
        #print("in loss:{}".format(torch.cuda.memory_allocated(0)))
        x, y, z = scores
        loss = nn.CrossEntropyLoss()
        '''t = torch.sum(t)/len(t)'''
        return loss(x, target) + loss(y, target)+ loss(z, target)
        #return self.bce(x, target) + self.bce(y, target) + self.bce(z, target)
    def neg_loss(self, scores, neg_label):
        x, y, z = scores
        loss = nn.CrossEntropyLoss()
        LOSS = torch.zeros(neg_label.shape[0])
        for i in range(neg_label.shape[1]):
            LOSS = loss(x, neg_label[:, i]) + loss(y, neg_label[:, i]) + loss(z, neg_label[:, i])
        
        return LOSS/neg_label.shape[1]


class LorenTzE_core(nn.Module):
    def __init__(self, embedding_dim, entities_num, relations_num, timestamps_num, drop_out, c=1.0, alpha=0.5, bias=-1):
        super(LorenTzE_core, self).__init__()
        self.embedding_dim = embedding_dim
        self.entities_num = entities_num
        self.relations_num = relations_num
        self.timestamps_num = timestamps_num

        #self.T_ct = nn.Embedding(self.timestamps_num, self.embedding_dim)
        self.time_mat = nn.Parameter(torch.zeros([self.timestamps_num, self.embedding_dim, self.embedding_dim]))

        self.cores = nn.Embedding(self.entities_num, self.embedding_dim)

        self.E_x = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_y = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_z = nn.Embedding(self.entities_num, self.embedding_dim)

        # self.R_r = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_x = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_y = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_z = nn.Embedding(self.relations_num, self.embedding_dim)

        self.E_bn = nn.BatchNorm1d(self.embedding_dim)

        self.b_x = nn.Parameter(torch.zeros(entities_num))
        self.b_y = nn.Parameter(torch.zeros(entities_num))
        self.b_z = nn.Parameter(torch.zeros(entities_num))
        self.b_ct = nn.Parameter(torch.zeros(entities_num))
        #self.time_trans = nn.Linear(embedding_dim*2, embedding_dim)

        # self.c_embedding = nn.Embedding(1, self.embedding_dim)
        self.drop_out = nn.Dropout(drop_out)
        self.c = c
        self.alpha = alpha
        self.bias = bias
        self.bce = nn.BCELoss()

    def init(self):
        #nn.init.xavier_normal_(self.T_ct.weight.data)
        nn.init.xavier_normal_(self.E_x.weight.data)
        nn.init.xavier_normal_(self.E_y.weight.data)
        nn.init.xavier_normal_(self.E_z.weight.data)
        nn.init.xavier_normal_(self.R_x.weight.data)
        nn.init.xavier_normal_(self.R_y.weight.data)
        nn.init.xavier_normal_(self.R_z.weight.data)
        nn.init.xavier_normal_(self.cores.weight.data)

    def forward(self, head, rel, timestamp, target):
        '''head = head.view(-1)
        rel = rel.view(-1)
        timestamp = timestamp.view(-1)'''

        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)

        #time = self.T_ct(timestamp)
        time = self.time_mat[timestamp]
        core_time = self.cores(head)
        h_ct = torch.bmm(time, core_time.unsqueeze(2)).squeeze()


        h_ct = self.E_bn(h_ct)
        h_x = self.E_bn(h_x)
        h_y = self.E_bn(h_y)
        h_z = self.E_bn(h_z)

        r_x = self.R_x(rel)
        r_y = self.R_y(rel)
        r_z = self.R_z(rel)
        length = torch.sqrt(r_x ** 2 + r_y ** 2 + r_z ** 2).detach()

        #h_ct = torch.sin(h_ct)

        r_v_rate = torch.sigmoid(length)

        r_x = r_x / length
        r_y = r_y / length
        r_z = r_z / length

        r_v_rate_2 = r_v_rate * r_v_rate
        gamma = 1.0 / (torch.sqrt(1.0 - r_v_rate_2))

        t_ct = gamma * h_ct + gamma * r_x * r_v_rate * h_x + gamma * r_y * r_v_rate * h_y + gamma * r_z * r_v_rate * h_z

        t_x = r_x * r_v_rate * gamma * h_ct + \
              (1 + (r_x * r_x * (gamma - 1))) * h_x + \
              (r_x * r_y * (gamma - 1)) * h_y + \
              (r_x * r_z * (gamma - 1)) * h_z

        t_y = r_y * r_v_rate * gamma * h_ct + \
              (r_x * r_y * (gamma - 1)) * h_x + \
              (1 + (r_y * r_y * (gamma - 1))) * h_y + \
              (r_z * r_y * (gamma - 1)) * h_z

        t_z = r_z * r_v_rate * gamma * h_ct + \
              (r_x * r_z * (gamma - 1)) * h_x + \
              (r_y * r_z * (gamma - 1)) * h_y + \
              (1 + (r_z * r_z * (gamma - 1))) * h_z

        t_ct = self.drop_out(t_ct)
        t_x = self.drop_out(t_x)
        t_y = self.drop_out(t_y)
        t_z = self.drop_out(t_z)

        score_x = torch.bmm(t_x.unsqueeze(1), self.E_x(target).unsqueeze(2)).squeeze()
        score_y = torch.bmm(t_y.unsqueeze(1), self.E_y(target).unsqueeze(2)).squeeze()
        score_z = torch.bmm(t_z.unsqueeze(1), self.E_z(target).unsqueeze(2)).squeeze()
        '''print(time.shape)
        temp = time.expand(self.entities_num, self.embedding_dim)
        print(temp.shape)'''
        ct_measure = torch.bmm(time, self.cores(target).unsqueeze(2)).squeeze()
        score_ct = torch.bmm(t_ct.unsqueeze(1), ct_measure.unsqueeze(2)).squeeze()
        #score_ct = torch.bmm(t_ct.unsqueeze(2).transpose(2, 1), ct_measure.unsqueeze(2)).squeeze()
        #print(score_ct.shape)
        #score_ct = torch.mm(t_ct, self.cores.weight.transpose(1,0))

        '''score_x = score_x + self.b_x.expand_as(score_x)
        score_y = score_y + self.b_y.expand_as(score_y)
        score_z = score_z + self.b_z.expand_as(score_z)
        score_ct = score_ct + self.b_ct.expand_as(score_ct)'''

        '''score_x = torch.sigmoid(torch.mm(t_x, self.E_x.weight.transpose(1, 0)))
        score_y = torch.sigmoid(torch.mm(t_y, self.E_y.weight.transpose(1, 0)))
        score_z = torch.sigmoid(torch.mm(t_z, self.E_z.weight.transpose(1, 0)))'''
        # print("before mm:{}".format(torch.cuda.memory_allocated(0)))
        # print("after mm:{}".format(torch.cuda.memory_allocated(0)))
        '''t_score = torch.sum((h_ct - t_ct)**2, dim=1)
        t_score = F.relu(t_score + self.bias)'''

        return score_x, score_y, score_z, score_ct

    def loss(self, scores):
        # print("in loss:{}".format(torch.cuda.memory_allocated(0)))
        score_x, score_y, score_z, score_ct= scores
        score_x = -F.logsigmoid(score_x).mean()
        score_y = -F.logsigmoid(score_y).mean()
        score_z = -F.logsigmoid(score_z).mean()
        score_ct = -F.logsigmoid(score_ct).mean()
        '''t = torch.sum(t)/len(t)'''
        return score_x + score_y + score_z + score_ct
        # return self.bce(x, target) + self.bce(y, target) + self.bce(z, target)

    def neg_loss(self, head, rel, timestamp, neg_label):
        LOSS = torch.zeros(neg_label.shape[0]).to('cuda')
        for i in range(neg_label.shape[1]):
            neg_x, neg_y, neg_z, neg_ct = self.forward(head, rel, timestamp, neg_label[:, i])
            neg_x = -F.logsigmoid(-neg_x)
            neg_y = -F.logsigmoid(-neg_y)
            neg_z = -F.logsigmoid(-neg_z)
            neg_ct = -F.logsigmoid(-neg_ct)
            temp = neg_x + neg_y + neg_z + neg_ct
            LOSS += temp

        LOSS /= neg_label.shape[1]
        return LOSS.mean()







class LorenTzE_history(nn.Module):
    def __init__(self, embedding_dim, entities_num, relations_num, timestamps_num, drop_out, c=1.0, alpha = 0.5):
        super(LorenTzE_history, self).__init__()
        self.embedding_dim = embedding_dim
        self.entities_num = entities_num
        self.relations_num = relations_num
        self.timestamps_num = timestamps_num

        self.T_ct = nn.Embedding(self.timestamps_num, self.embedding_dim)

        self.E_x = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_y = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_z = nn.Embedding(self.entities_num, self.embedding_dim)

        # self.R_r = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_x = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_y = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_z = nn.Embedding(self.relations_num, self.embedding_dim)

        self.T_bn = nn.BatchNorm1d(self.embedding_dim)
        self.E_bn = nn.BatchNorm1d(self.embedding_dim)
        self.R_bn = nn.BatchNorm1d(self.embedding_dim)
        self.b_a = nn.Parameter(torch.zeros(entities_num))
        self.b_x = nn.Parameter(torch.zeros(entities_num))
        self.b_y = nn.Parameter(torch.zeros(entities_num))
        self.b_z = nn.Parameter(torch.zeros(entities_num))

        # self.c_embedding = nn.Embedding(1, self.embedding_dim)
        self.drop_out = nn.Dropout(drop_out)
        self.c = c
        self.alpha = alpha
        self.bce = nn.BCELoss()
        # self.bias = 0.01

    def init(self):
        nn.init.xavier_normal_(self.T_ct.weight.data)
        nn.init.xavier_normal_(self.E_x.weight.data)
        nn.init.xavier_normal_(self.E_y.weight.data)
        nn.init.xavier_normal_(self.E_z.weight.data)
        nn.init.xavier_normal_(self.R_x.weight.data)
        nn.init.xavier_normal_(self.R_y.weight.data)
        nn.init.xavier_normal_(self.R_z.weight.data)
        # nn.init.xavier_normal_(self.R_r.weight.data)
        # nn.init.xavier_normal_(self.R_theta.weight.data)
        # nn.init.xavier_normal_(self.R_phi.weight.data)
    def fact_predict(self, head, rel, timestamp):
        head = head.view(-1)
        rel = rel.view(-1)
        timestamp = timestamp.view(-1)
        h_ct = self.E_ct(timestamp)

        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)

        h_ct = self.E_bn(h_ct)
        h_x = self.E_bn(h_x)
        h_y = self.E_bn(h_y)
        h_z = self.E_bn(h_z)
        # print("h:", h_ct.shape, h_x.shape, h_y.shape, h_z.shape)

        # r_r = self.R_r(rel)
        r_x = self.R_x(rel)
        r_y = self.R_y(rel)
        r_z = self.R_z(rel)
        length = torch.sqrt(r_x ** 2 + r_y ** 2 + r_z ** 2).detach()
        # r_theta = self.R_theta(rel)
        # r_phi = self.R_phi(rel)
        # print("r:", r_r.shape, r_theta.shape, r_phi.shape)

        r_v_rate = torch.sigmoid(length)
        # r_v = r_v_rate * self.c

        r_x = r_x / length
        r_y = r_y / length
        r_z = r_z / length

        r_v_rate_2 = r_v_rate * r_v_rate
        gamma = 1.0 / (torch.sqrt(1.0 - r_v_rate_2))

        t_ct = gamma * h_ct + gamma * r_x * r_v_rate * h_x + gamma * r_y * r_v_rate * h_y + gamma * r_z * r_v_rate * h_z

        t_x = r_x * r_v_rate * gamma * h_ct + \
              (1 + (r_x * r_x * (gamma - 1))) * h_x + \
              (r_x * r_y * (gamma - 1)) * h_y + \
              (r_x * r_z * (gamma - 1)) * h_z

        t_y = r_y * r_v_rate * gamma * h_ct + \
              (r_x * r_y * (gamma - 1)) * h_x + \
              (1 + (r_y * r_y * (gamma - 1))) * h_y + \
              (r_z * r_y * (gamma - 1)) * h_z

        t_z = r_z * r_v_rate * gamma * h_ct + \
              (r_x * r_z * (gamma - 1)) * h_x + \
              (r_y * r_z * (gamma - 1)) * h_y + \
              (1 + (r_z * r_z * (gamma - 1))) * h_z
        # print("t:", t_ct.shape, t_x.shape, t_y.shape, t_z.shape)

        t_ct = self.drop_out(t_ct)

        t_x = self.drop_out(t_x)
        t_y = self.drop_out(t_y)
        t_z = self.drop_out(t_z)
        return t_x, t_y, t_z



    def time_predict(self, head, rel, history):
        head = head.view(-1)
        rel = rel.view(-1)

        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)


        return t_x, t_y, t_z, t_time




    def forward(self, head, rel, timestamp):
        fact_x, fact_y, fact_z = self.fact_predict(head, rel, timestamp)

        time_x, time_y, time_z, time_t = self.time_predict(head, rel, )

        T_ct = self.drop_out(self.E_bn(self.T_ct.weight))
        E_x = self.drop_out(self.E_bn(self.E_x.weight))
        E_y = self.drop_out(self.E_bn(self.E_y.weight))
        E_z = self.drop_out(self.E_bn(self.E_z.weight))

        score_ct = torch.mm(t_ct, T_ct.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_x = torch.mm(t_x, E_x.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_y = torch.mm(t_y, E_y.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_z = torch.mm(t_z, E_z.transpose(1, 0))  # Bxd, Exd -> BxE,


        score_ct = torch.sigmoid(score_ct + self.b_a.expand_as(score_ct))
        score_x = torch.sigmoid(score_x + self.b_x.expand_as(score_x))
        score_y = torch.sigmoid(score_y + self.b_y.expand_as(score_y))
        score_z = torch.sigmoid(score_z + self.b_z.expand_as(score_z))
        # score = (score_ct + score_x + score_y + score_z) / 4
        return score_ct, score_x, score_y, score_z

    def loss(self, target, y):
        y_a, y_ai, y_b, y_bi = target
        return self.bce(y_a, y) + self.bce(y_ai, y) + self.bce(y_b, y) + self.bce(y_bi, y)

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

        # self.R_r = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_x = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_y = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_z = nn.Embedding(self.relations_num, self.embedding_dim)

        self.E_bn = nn.BatchNorm1d(self.embedding_dim)
        self.R_bn = nn.BatchNorm1d(self.embedding_dim)
        self.b_a = nn.Parameter(torch.zeros(entities_num))
        self.b_x = nn.Parameter(torch.zeros(entities_num))
        self.b_y = nn.Parameter(torch.zeros(entities_num))
        self.b_z = nn.Parameter(torch.zeros(entities_num))

        # self.c_embedding = nn.Embedding(1, self.embedding_dim)
        self.drop_out = nn.Dropout(drop_out)
        self.c = c
        self.bce = nn.BCELoss()
        # self.bias = 0.01

    def init(self):
        nn.init.xavier_normal_(self.E_ct.weight.data)
        nn.init.xavier_normal_(self.E_x.weight.data)
        nn.init.xavier_normal_(self.E_y.weight.data)
        nn.init.xavier_normal_(self.E_z.weight.data)
        nn.init.xavier_normal_(self.R_x.weight.data)
        nn.init.xavier_normal_(self.R_y.weight.data)
        nn.init.xavier_normal_(self.R_z.weight.data)
        # nn.init.xavier_normal_(self.R_r.weight.data)
        # nn.init.xavier_normal_(self.R_theta.weight.data)
        # nn.init.xavier_normal_(self.R_phi.weight.data)

    def forward(self, head, rel):
        head = head.view(-1)
        rel = rel.view(-1)
        h_ct = self.E_ct(head)
        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)

        h_ct = self.E_bn(h_ct)
        h_x = self.E_bn(h_x)
        h_y = self.E_bn(h_y)
        h_z = self.E_bn(h_z)
        # print("h:", h_ct.shape, h_x.shape, h_y.shape, h_z.shape)

        # r_r = self.R_r(rel)
        r_x = self.R_x(rel)
        r_y = self.R_y(rel)
        r_z = self.R_z(rel)
        length = torch.sqrt(r_x **2 + r_y**2 + r_z**2).detach()
        # r_theta = self.R_theta(rel)
        # r_phi = self.R_phi(rel)
        # print("r:", r_r.shape, r_theta.shape, r_phi.shape)

        r_v_rate = torch.sigmoid(length)
        # r_v = r_v_rate * self.c

        r_x = r_x / length
        r_y = r_y / length
        r_z = r_z / length

        r_v_rate_2 = r_v_rate * r_v_rate
        gamma = 1.0 / (torch.sqrt(1.0 - r_v_rate_2))

        t_ct = gamma * h_ct + gamma * r_x * r_v_rate * h_x + gamma * r_y * r_v_rate * h_y + gamma * r_z * r_v_rate * h_z

        t_x = r_x * r_v_rate * gamma * h_ct + \
              (1 + (r_x * r_x * (gamma - 1))) * h_x + \
              (r_x * r_y * (gamma - 1)) * h_y + \
              (r_x * r_z * (gamma - 1)) * h_z

        t_y = r_y * r_v_rate * gamma * h_ct + \
              (r_x * r_y * (gamma - 1)) * h_x + \
              (1 + (r_y * r_y * (gamma - 1))) * h_y + \
              (r_z * r_y * (gamma - 1)) * h_z

        t_z = r_z * r_v_rate * gamma * h_ct + \
              (r_x * r_z * (gamma - 1)) * h_x + \
              (r_y * r_z * (gamma - 1)) * h_y + \
              (1 + (r_z * r_z * (gamma - 1))) * h_z
        # print("t:", t_ct.shape, t_x.shape, t_y.shape, t_z.shape)

        t_ct = self.drop_out(t_ct)
        t_x = self.drop_out(t_x)
        t_y = self.drop_out(t_y)
        t_z = self.drop_out(t_z)

        E_ct = self.drop_out(self.E_bn(self.E_ct.weight))
        E_x = self.drop_out(self.E_bn(self.E_x.weight))
        E_y = self.drop_out(self.E_bn(self.E_y.weight))
        E_z = self.drop_out(self.E_bn(self.E_z.weight))

        score_ct = torch.mm(t_ct, E_ct.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_x = torch.mm(t_x, E_x.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_y = torch.mm(t_y, E_y.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_z = torch.mm(t_z, E_z.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_ct = torch.sigmoid(score_ct + self.b_a.expand_as(score_ct))
        score_x = torch.sigmoid(score_x + self.b_x.expand_as(score_x))
        score_y = torch.sigmoid(score_y + self.b_y.expand_as(score_y))
        score_z = torch.sigmoid(score_z + self.b_z.expand_as(score_z))
        # score = (score_ct + score_x + score_y + score_z) / 4
        return score_ct, score_x, score_y, score_z

    def loss(self, target, y):
        y_a, y_ai, y_b, y_bi = target
        return self.bce(y_a, y) + self.bce(y_ai, y) + self.bce(y_b, y) + self.bce(y_bi, y)

class Lorentz_tedius(nn.Module):

    def __init__(self, embedding_dim, entities_num, relations_num, drop_out, c = 1.0):
        super(Lorentz_tedius, self).__init__()
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



    def forward(self, head, rel):
        h_ct = self.E_ct(head)
        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)

        r_x = self.R_vx(rel)
        r_y = self.R_vy(rel)
        r_z = self.R_vz(rel)

        r_v = r_x * r_x +  r_y * r_y +  r_z * r_z
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

    def __init__(self, embedding_dim, entities_num, relations_num, drop_out, c = 1.0):
        super(Lorentz_sphere, self).__init__()
        self.embedding_dim = embedding_dim
        self.entities_num = entities_num
        self.relations_num = relations_num

        self.E_ct = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_x = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_y = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_z = nn.Embedding(self.entities_num, self.embedding_dim)

        #self.R_r = nn.Embedding(self.relations_num, self.embedding_dim, max_norm=1)
        self.R_r = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_theta = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_phi = nn.Embedding(self.relations_num, self.embedding_dim)
        #self.c_embedding = nn.Embedding(1, self.embedding_dim)
        self.drop_out = nn.Dropout(drop_out)
        self.c = c
        self.loss = nn.BCELoss()
       # self.bias = 0.01

    def forward(self, head, rel):
        h_ct = self.E_ct(head)
        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)
        #print('rel_in: ', rel)
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


        #print(self.E_ct.weight.transpose(1, 0).shape)
        #print(h_ct.shape)
        #print('h_ct:    ', h_ct)
        #print('t_ct:    ', t_ct)
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
        #print('t_shape: ', t_ct.shape)
        score_ct = torch.mm(t_ct.squeeze(1), self.E_ct.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_x = torch.mm(t_x.squeeze(1), self.E_x.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_y = torch.mm(t_y.squeeze(1), self.E_y.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_z = torch.mm(t_z.squeeze(1), self.E_z.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score = (score_ct + score_x + score_y + score_z) / 4

        #print(score)
        score = torch.sigmoid(score)
        return score
        #return (t_ct, t_x, t_y, t_z)

class Lorentz_changeable(nn.Module):

    def __init__(self, embedding_dim, entities_num, relations_num, drop_out, c = 1.0):
        super(Lorentz_changeable, self).__init__()
        self.embedding_dim = embedding_dim
        self.entities_num = entities_num
        self.relations_num = relations_num

        self.E_ct = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_x = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_y = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_z = nn.Embedding(self.entities_num, self.embedding_dim)

        #self.R_r = nn.Embedding(self.relations_num, self.embedding_dim, max_norm=1)
        self.R_r = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_theta = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_phi = nn.Embedding(self.relations_num, self.embedding_dim)
        self.c_embedding = nn.Parameter(torch.FloatTensor(1, embedding_dim))
        nn.init.xavier_normal_(self.c_embedding)
        self.drop_out = nn.Dropout(drop_out)
        #self.c = c
        self.loss = nn.BCELoss()
       # self.bias = 0.01

    def forward(self, head, rel):
        h_ct = self.E_ct(head)
        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)
        #print('rel_in: ', rel)
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


        #print(self.E_ct.weight.transpose(1, 0).shape)
        #print(h_ct.shape)
        #print('h_ct:    ', h_ct)
        #print('t_ct:    ', t_ct)
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
        #print('t_shape: ', t_ct.shape)
        score_ct = torch.mm(t_ct.squeeze(1), self.E_ct.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_x = torch.mm(t_x.squeeze(1), self.E_x.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_y = torch.mm(t_y.squeeze(1), self.E_y.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_z = torch.mm(t_z.squeeze(1), self.E_z.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score = (score_ct + score_x + score_y + score_z) / 4

        #print(score)
        score = torch.sigmoid(score)
        return score
        #return (t_ct,


class Lorentz_multi_speed(nn.Module):

    def __init__(self, embedding_dim, entities_num, relations_num, drop_out, c = 1.0):
        super(Lorentz_multi_speed, self).__init__()
        self.embedding_dim = embedding_dim
        self.entities_num = entities_num
        self.relations_num = relations_num

        self.E_ct = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_x = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_y = nn.Embedding(self.entities_num, self.embedding_dim)
        self.E_z = nn.Embedding(self.entities_num, self.embedding_dim)

        #self.R_r = nn.Embedding(self.relations_num, self.embedding_dim, max_norm=1)
        self.R_r = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_theta = nn.Embedding(self.relations_num, self.embedding_dim)
        self.R_phi = nn.Embedding(self.relations_num, self.embedding_dim)
        #self.c_embedding = nn.Parameter(torch.FloatTensor(1, embedding_dim))
        self.c_embedding = nn.Embedding(self.relations_num, self.embedding_dim)
        nn.init.xavier_normal_(self.c_embedding.weight.data)
        self.drop_out = nn.Dropout(drop_out)
        #self.c = c
        self.loss = nn.BCELoss()
       # self.bias = 0.01

    def forward(self, head, rel):
        h_ct = self.E_ct(head)
        h_x = self.E_x(head)
        h_y = self.E_y(head)
        h_z = self.E_z(head)
        #print('rel_in: ', rel)
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


        #print(self.E_ct.weight.transpose(1, 0).shape)
        #print(h_ct.shape)
        #print('h_ct:    ', h_ct)
        #print('t_ct:    ', t_ct)
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
        #print('t_shape: ', t_ct.shape)
        score_ct = torch.mm(t_ct.squeeze(1), self.E_ct.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_x = torch.mm(t_x.squeeze(1), self.E_x.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_y = torch.mm(t_y.squeeze(1), self.E_y.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score_z = torch.mm(t_z.squeeze(1), self.E_z.weight.transpose(1, 0))  # Bxd, Exd -> BxE,
        score = (score_ct + score_x + score_y + score_z) / 4

        #print(score)
        score = torch.sigmoid(score)
        return score


