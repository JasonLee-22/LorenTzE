import numpy as np
import torch
import os

def load_quadruples(file_name):
    f = open(file_name, 'r')
    quadruples = []
    timestamps = set()
    for line in f.readlines():
        ls = line.strip().split()
        head, rel, tail, timestamp = ls[0], ls[1], ls[2], ls[3]
        quadruples.append([int(head), int(rel), int(tail), int(timestamp)/24])
        timestamps.add(int(timestamp))
    timestamps = list(timestamps)
    timestamps.sort()

    return quadruples, timestamps

def filter_test(h_or_t, h, r, t, entities_num, to_be_filtered):
    candidates = []
    if h_or_t == 't':
        h, r, t = int(h), int(r), int(t)
        #print(h,r,t)
        if (h, r, t) in to_be_filtered:
            to_be_filtered.remove((h, r, t))
        for candidate_ts in range(entities_num):
            if (h, r, candidate_ts) not in to_be_filtered:
                candidates.append(candidate_ts)
    elif h_or_t == 'h':
        h, r, t = int(h), int(r), int(t)
        if (h, r, t) in to_be_filtered:
            to_be_filtered.remove((h, r, t))
        for candidate_hs in range(entities_num):
            if (candidate_hs, r, t) not in to_be_filtered:
                candidates.append(candidate_hs)
    return torch.LongTensor(candidates)

def ranking(h_or_t, entities_num, scores, batch_size, h, r, t, to_be_filtered):
    ranks = []
    if h_or_t == 't':
        for i in range(batch_size):
            head, rel, tail = h[i], r[i], t[i]
            candidates = filter_test(h_or_t, head, rel, tail, entities_num, to_be_filtered)
            #print('can: ', candidates.device)
            #print('tail: ', tail.device)
            true_index = torch.nonzero((candidates == tail))
            true_index = true_index.squeeze()
            #print('true: ', true_index)
            _, rank_index = torch.sort(scores[i][candidates], descending=True)
            #print('rank_index: ', rank_index)
            rank = torch.nonzero((rank_index == true_index))
            rank = rank.squeeze()
            #print('rank: ', rank)
            rank += 1
            #print(rank)
            ranks.append(rank)
    if h_or_t == 'h':
         for i in range(batch_size):
            head, rel, tail = h[i], r[i], t[i]
            candidates = filter_test(h_or_t, head, rel, tail, entities_num, to_be_filtered)
            true_index = torch.nonzero(int(candidates == head))
            true_index = true_index.squeeze()
            _, rank_index = torch.sort(scores[i][candidates], descending=True)
            rank = torch.nonzero(int(rank_index == true_index))
            rank = rank.squeeze()
            rank += 1
            ranks.append(rank)
    return torch.Tensor(ranks)

def metric(h_or_t, entities_num, scores, to_be_filtered, test_data, batch_size, hits = [1, 3, 10]):
    with torch.no_grad():
        head = test_data[:, 0]
        rel = test_data[:, 1]
        tail = test_data[:, 2]
        #scores = scores[0] + scores[1] + scores[2]
        ranks = ranking(h_or_t, entities_num, scores, batch_size, head, rel, tail, to_be_filtered)
        mrr = torch.mean(1.0 / ranks)
        mr = torch.mean(ranks)
        hits1 = torch.sum(ranks <= hits[0])
        hits3 = torch.sum(ranks <= hits[1])
        hits10 = torch.sum(ranks <= hits[2])

        return mr.item(), mrr.item(), hits1.item(), hits3.item(), hits10.item()



