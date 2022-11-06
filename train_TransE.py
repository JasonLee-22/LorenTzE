import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
from utils_static import *
from main import args
import os
class TransE(nn.Module):
    def __init__(self, embedding_dim, entities_num, relations_num, drop_out):
        super(TransE, self).__init__()
        self.embedding_dim = embedding_dim
        self.entities_num = entities_num
        self.relations_num = relations_num
        self.drop_out = nn.Dropout(drop_out)
        self.E = nn.Embedding(self.entities_num, self.embedding_dim)
        self.R = nn.Embedding(self.relations_num, self.embedding_dim)

    def init(self):
        nn.init.xavier_normal_(self.E.weight.data)
        nn.init.xavier_normal_(self.R.weight.data)

    def forward(self, head, rel):
        head = self.E(head)
        rel = self.R(rel)
        tail = F.relu(head + rel)
        score = torch.mm(tail, self.E.weight.transpose(0,1))
        return score

    def loss(self, scores, target):
        #print("in loss:{}".format(torch.cuda.memory_allocated(0)))
        loss = nn.CrossEntropyLoss()
        return loss(scores, target)



dataset = args.dataset
if dataset == 'ICEWS14':
    train, train_times = load_quadruples('./data/ICEWS14/train.txt')
    valid, valid_times = load_quadruples('./data/ICEWS14/valid.txt')
    test, test_times = load_quadruples('./data/ICEWS14/test.txt')
    stat = open('./data/ICEWS14/stat.txt', 'r')


epochs = args.epochs
batch_size = args.batch_size
dropout = args.dropout
lr = args.lr
h_or_t = args.h_or_t
name = args.name
if not os.path.exists('./output'):
    os.mkdir('./output')
if not os.path.exists('./output/' + dataset):
    os.mkdir('./output/' + dataset)
if not os.path.exists('./output/' + dataset + '/' + name):
    os.mkdir('./output/' + dataset + '/' + name)

fout = open('./output/' + dataset + '/' + name + '/' + 'output_' + name, 'w')

best_mrr = 0
best_mr = 0
best_hits1 = 0
best_hits3 = 0
best_hits10 = 0

embedding_dim = args.dim


entity_num, relation_num, time_num = 0,0,0
for i in stat.readlines():
    entity_num, relation_num, time_num = i.split('\t')

entity_num = int(entity_num)
relation_num = int(relation_num)
time_num = int(time_num)


model = TransE(embedding_dim, entity_num, relation_num, dropout)
model.to('cuda')
model.init()
train = np.asarray(train)
valid = np.asarray(valid)
test = np.asarray(test)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
best_epoch = -1
for i in range(epochs):
    model.train()
    start = 0
    end = batch_size
    total_loss = 0
    while end <= train.shape[0]:
        if h_or_t == 't':
            labels = torch.LongTensor(train[start:end, 2]).to('cuda')
            inputs = torch.LongTensor(train[start:end, 0]).to('cuda')
            rels = torch.LongTensor(train[start:end, 1]).to('cuda')
            ts = torch.LongTensor(train[start:end, 3]).to('cuda')
        else:
            labels = torch.LongTensor(train[start:end, 0]).to('cuda')
            inputs = torch.LongTensor(train[start:end, 2]).to('cuda')
            rels = torch.LongTensor(train[start:end, 1]).to('cuda')
            ts = torch.LongTensor(train[start:end, 3]).to('cuda')

        scores = model.forward(inputs, rels)
        loss = model.loss(scores, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        print('Batch: {} \t Batch Loss: {}'.format(start//batch_size, loss))
        if end == train.shape[0]:
            break
        else:
            start = end
            end = min(end + batch_size, train.shape[0])

    print('Epoch: {},  AVG_Loss: {}'.format(i, total_loss / (int(train.shape[0]/batch_size) + 1)))
    fout.write('Epoch: {},  AVG_Loss: {}\n'.format(i, total_loss / (int(train.shape[0]/batch_size) + 1)))


    #validation:
    model.eval()
    v_start = 0
    v_end = batch_size
    v_loss = 0
    mr = 0
    mrr = 0
    hits1 = 0
    hits3 = 0
    hits10 = 0
    to_be_filtered = [(i[0], i[1], i[2]) for i in train]
    #print(len(to_be_filtered))
    to_be_filtered += [(i[0], i[1], i[2]) for i in valid]
    to_be_filtered = set(to_be_filtered)
    #print(len(to_be_filtered))
    while v_end <= valid.shape[0]:
        if h_or_t == 't':
            labels = torch.LongTensor(valid[v_start:v_end, 2]).to('cuda')
            inputs = torch.LongTensor(valid[v_start:v_end, 0]).to('cuda')
            rels = torch.LongTensor(valid[v_start:v_end, 1]).to('cuda')
            ts = torch.LongTensor(valid[v_start:v_end, 3]).to('cuda')
        else:
            labels = torch.LongTensor(valid[v_start:v_end, 0]).to('cuda')
            inputs = torch.LongTensor(valid[v_start:v_end, 2]).to('cuda')
            rels = torch.LongTensor(valid[v_start:v_end, 1]).to('cuda')
            ts = torch.LongTensor(valid[v_start:v_end, 3]).to('cuda')

        #print("before for:{}".format(torch.cuda.memory_allocated(0)))
        scores = model.forward(inputs, rels)
        #print("after for:{}".format(torch.cuda.memory_allocated(0)))
        loss = model.loss(scores, labels)
        #print("after loss:{}".format(torch.cuda.memory_allocated(0)))
        v_loss += loss.item()
        #scores = scores[0] + scores[1] + scores[2]
        #print(v_start, v_end, valid.shape)
        v_mr, v_mrr, v_hits1, v_hits3, v_hits10 = metric(h_or_t, entity_num, scores, to_be_filtered, torch.LongTensor(valid[v_start:v_end]), v_end- v_start)
        #print("after metric:{}".format(torch.cuda.memory_allocated(0)))
        mr += v_mr * (v_end- v_start)
        mrr += v_mrr * (v_end - v_start)
        hits1 += v_hits1
        hits3 += v_hits3
        hits10 += v_hits10
        if v_end == valid.shape[0]:
            break
        else:
            v_start = v_end
            v_end = min(v_end + batch_size, valid.shape[0])

    mr /= valid.shape[0]
    mrr /= valid.shape[0]
    hits1 /= valid.shape[0]
    hits3 /= valid.shape[0]
    hits10 /= valid.shape[0]

    if mrr > best_mrr:
        best_mrr = mrr
        torch.save(model, './output/' + dataset + '/' + name + '/best_epoch_{}.ckpt'.format(i))
        best_epoch = i
    if hits1 > best_hits1:
        best_hits1 = hits1
    if hits3 > best_hits3:
        best_hits3 = hits3
    if hits10 > best_hits10:
        best_hits10 = hits10

    print('Epoch: {} \nValidation: Loss: {} \tMRR: {} \tHits@1: {} \tHits@3: {}\tHits@10: {}'.format(i, v_loss, mrr, hits1, hits3, hits10))
    fout.write('Epoch: {} \nValidation: Loss: {} \tMRR: {} \tHits@1: {} \tHits@3: {}\tHits@10: {}\n'.format(i, v_loss, mrr, hits1, hits3, hits10))
