import torch
import numpy as np
from models import LorenTzE, LorenTzE_core
from models import LorenTzE_history
from utils import *
from main import args
import os
from torch.optim.lr_scheduler import StepLR as StepLR



def test_step(model, train, valid, test, batch_size, fout):
    # Test:
    model.eval()
    t_start = 0
    t_end = batch_size
    mr = 0
    mrr = 0
    hits1 = 0
    hits3 = 0
    hits10 = 0
    test_filtered = [(i[0], i[1], i[2], i[3]) for i in train]
    test_filtered += [(i[0], i[1], i[2], i[3]) for i in valid]
    test_filtered += [(i[0], i[1], i[2], i[3]) for i in test]
    test_filtered = set(test_filtered)
    while t_end <= test.shape[0]:
        if h_or_t == 't':
            labels = torch.LongTensor(test[t_start:t_end, 2]).to('cuda')
            inputs = torch.LongTensor(test[t_start:t_end, 0]).to('cuda')
            rels = torch.LongTensor(test[t_start:t_end, 1]).to('cuda')
            ts = torch.LongTensor(test[t_start:t_end, 3]).to('cuda')
        else:
            labels = torch.LongTensor(test[t_start:t_end, 0]).to('cuda')
            inputs = torch.LongTensor(test[t_start:t_end, 2]).to('cuda')
            rels = torch.LongTensor(test[t_start:t_end, 1]).to('cuda')
            ts = torch.LongTensor(test[t_start:t_end, 3]).to('cuda')

        test_score = model.forward(inputs, rels, ts, labels)
        #test_score = test_score[0] + test_score[1] + test_score[2]
        test_mr, test_mrr, test_hits1, test_hits3, test_hits10 = metric(h_or_t, entity_num, test_score, test_filtered,
                                                                        test[t_start: t_end], t_end - t_start)
        mr += test_mr * (t_end - t_start)
        mrr += test_mrr * (t_end - t_start)
        hits1 += test_hits1
        hits3 += test_hits3
        hits10 += test_hits10
        if t_end == test.shape[0]:
            break
        else:
            t_start = t_end
            t_end = min(t_end + batch_size, test.shape[0])
    print(hits10)
    print(test.shape[0])
    mr /= test.shape[0]
    mrr /= test.shape[0]
    hits1 /= test.shape[0]
    hits3 /= test.shape[0]
    hits10 /= test.shape[0]
    print(hits10)

    print('Test Result:  \n')
    print('MRR: {} \tHits@1: {} \tHits@3: {}\tHits@10: {}'.format(mrr, hits1, hits3, hits10))
    fout.write('Test Result:  \n')
    fout.write('MRR: {} \tHits@1: {} \tHits@3: {}\tHits@10: {}'.format(mrr, hits1, hits3, hits10))





dataset = args.dataset
'''if dataset == 'ICEWS14':
    train, train_times = load_quadruples('./data/ICEWS14/train.txt')
    valid, valid_times = load_quadruples('./data/ICEWS14/valid.txt')
    test, test_times = load_quadruples('./data/ICEWS14/test.txt')
    stat = open('./data/ICEWS14/stat.txt', 'r')
elif dataset == 'GDELT':
    train, train_times = load_quadruples('./data/GDELT/train.txt')
    valid, valid_times = load_quadruples('./data/GDELT/valid.txt')
    test, test_times = load_quadruples('./data/GDELT/test.txt')
    stat = open('./data/GDELT/stat.txt', 'r')
    print('GDELT')
    print(stat)'''

train, train_times = load_quadruples('./data/{}/train.txt'.format(dataset))
valid, valid_times = load_quadruples('./data/{}/valid.txt'.format(dataset))
test, test_times = load_quadruples('./data/{}/test.txt'.format(dataset))
stat = open('./data/{}/stat.txt'.format(dataset), 'r')


epochs = args.epochs
batch_size = args.batch_size
dropout = args.dropout
lr = args.lr
h_or_t = args.h_or_t
name = args.name
neg_size = args.negative_size

if not os.path.exists('./output'):
    os.mkdir('./output')
if not os.path.exists('./output/' + dataset):
    os.mkdir('./output/' + dataset)
if not os.path.exists('./output/' + dataset + '/' + name):
    os.mkdir('./output/' + dataset + '/' + name)

fout = open('./output/' + dataset + '/' + name + '/' + 'output_' + name + '.txt', 'w')

best_mrr = 0
best_mr = 0
best_hits1 = 0
best_hits3 = 0
best_hits10 = 0
best_epoch = -1

embedding_dim = args.dim


entity_num, relation_num, time_num = 0,0,0
for i in stat.readlines():
    entity_num, relation_num, time_num = i.split('\t')

entity_num = int(entity_num)
relation_num = int(relation_num)
time_num = int(time_num)


model = LorenTzE_core(embedding_dim, entity_num, relation_num, time_num, dropout)
model.to('cuda')
model.init()
train = np.asarray(train)
valid = np.asarray(valid)
test = np.asarray(test)


#scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for i in range(1, epochs + 1):
    model.train()
    if i >=300 and i % 10 == 0:
        lr = lr * 0.9
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()
    start = 0
    end = batch_size
    total_loss = 0
    while end <= train.shape[0]:
        if h_or_t == 't':
            labels = torch.LongTensor(train[start:end, 2]).to('cuda')
            inputs = torch.LongTensor(train[start:end, 0]).to('cuda')
            rels = torch.LongTensor(train[start:end, 1]).to('cuda')
            ts = torch.LongTensor(train[start:end, 3]).to('cuda')
            neg_tails = neg_sample(train[start:end], neg_size, h_or_t, entity_num)
            neg_labels = torch.LongTensor(neg_tails).to('cuda')
        else:
            labels = torch.LongTensor(train[start:end, 0]).to('cuda')
            inputs = torch.LongTensor(train[start:end, 2]).to('cuda')
            rels = torch.LongTensor(train[start:end, 1]).to('cuda')
            ts = torch.LongTensor(train[start:end, 3]).to('cuda')
            neg_heads = neg_sample(train[start:end], neg_size, h_or_t, entity_num)
            neg_labels = torch.LongTensor(neg_heads).to('cuda')


        scores = model.forward(inputs, rels, ts)
        #positive_loss = model.loss(scores, labels)
        positive_loss = model.loss(scores, labels, ts)
        #print('pos: ', positive_loss)
        '''negative_score = model.neg_loss(scores, neg_labels)
        negative_loss = model.loss(negative_score)'''
        negative_loss = model.neg_loss(inputs, rels, ts, neg_labels)
        #print('neg: ', negative_loss)
        loss = positive_loss + negative_loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        #scheduler.step()
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
    to_be_filtered = [(i[0], i[1], i[2], i[3]) for i in train]
    #print(len(to_be_filtered))
    to_be_filtered += [(i[0], i[1], i[2], i[3]) for i in valid]
    to_be_filtered += [(i[0], i[1], i[2], i[3]) for i in test]
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
        scores = model.forward(inputs, rels, ts)
        #print("after for:{}".format(torch.cuda.memory_allocated(0)))
        #loss = model.loss(scores, labels)
        loss = model.loss(scores, labels, ts)
        #print("after loss:{}".format(torch.cuda.memory_allocated(0)))
        v_loss += loss.item()
        #scores = scores[0] + scores[1] + scores[2]
        #print(v_start, v_end, valid.shape)
        #v_mr, v_mrr, v_hits1, v_hits3, v_hits10 = metric(h_or_t, entity_num, scores, to_be_filtered, torch.LongTensor(valid[v_start:v_end]), v_end - v_start)
        v_mr, v_mrr, v_hits1, v_hits3, v_hits10 = specific_ranking(h_or_t=h_or_t, batch_size=v_end - v_start, entities_num=entity_num, to_be_filtered=to_be_filtered,
                                                         samples = torch.LongTensor(valid[v_start:v_end]), model = model)
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
    print('Current learning rate: {}\n'.format(lr))
    print('Epoch: {} \nValidation: Loss: {} \tMRR: {} \tHits@1: {} \tHits@3: {}\tHits@10: {}'.format(i, v_loss, mrr, hits1, hits3, hits10))
    fout.write('Current learning rate: {}\n'.format(lr))
    fout.write('Epoch: {} \nValidation: Loss: {} \tMRR: {} \tHits@1: {} \tHits@3: {}\tHits@10: {}\n'.format(i, v_loss, mrr, hits1, hits3, hits10))

    if i % 100 == 0:
        best_model = torch.load('./output/' + dataset + '/' + name + '/best_epoch_{}.ckpt'.format(best_epoch))
        print('Test at epoch{}: \n'.format(i))
        fout.write('Test at epoch{}: \n'.format(i))
        test_step(best_model, train, valid, test, batch_size, fout)

print('BEST: MRR: {} \tHits@1: {} \tHits@3: {}\tHits@10: {}'.format(best_mrr, best_hits1, best_hits3, best_hits10))
fout.write('BEST: MRR: {} \tHits@1: {} \tHits@3: {}\tHits@10: {}\n'.format(best_mrr, best_hits1, best_hits3, best_hits10))

