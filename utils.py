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
        if 'GDELT' in file_name:
            quadruples.append([int(head), int(rel), int(tail), int(timestamp) / 15])
            timestamps.add(int(timestamp) / 15)
        else:
            quadruples.append([int(head), int(rel), int(tail), int(timestamp)/24])
            timestamps.add(int(timestamp)/24)
    timestamps = list(timestamps)
    timestamps.sort()

    return quadruples, timestamps

def filter_test(h_or_t, h, r, t, time, entities_num, to_be_filtered):
    candidates = []
    if h_or_t == 't':
        h, r, t, time = int(h), int(r), int(t), int(time)
        #print(h,r,t)
        if (h, r, t, time) in to_be_filtered:
            to_be_filtered.remove((h, r, t, time))
        for candidate_ts in range(entities_num):
            if (h, r, candidate_ts, time) not in to_be_filtered:
                candidates.append(candidate_ts)
    elif h_or_t == 'h':
        h, r, t, time = int(h), int(r), int(t), int(time)
        if (h, r, t) in to_be_filtered:
            to_be_filtered.remove((h, r, t, time))
        for candidate_hs in range(entities_num):
            if (candidate_hs, r, t, time) not in to_be_filtered:
                candidates.append(candidate_hs)
    return torch.LongTensor(candidates)

def ranking(h_or_t, entities_num, scores, batch_size, h, r, t, time, to_be_filtered):
    ranks = []
    if h_or_t == 't':
        for i in range(batch_size):
            head, rel, tail, ts = h[i], r[i], t[i], time[i]
            candidates = filter_test(h_or_t, head, rel, tail, ts, entities_num, to_be_filtered)
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
            head, rel, tail, ts = h[i], r[i], t[i], time[i]
            candidates = filter_test(h_or_t, head, rel, tail, ts, entities_num, to_be_filtered)
            true_index = torch.nonzero(int(candidates == head))
            true_index = true_index.squeeze()
            _, rank_index = torch.sort(scores[i][candidates], descending=True)
            rank = torch.nonzero(int(rank_index == true_index))
            rank = rank.squeeze()
            rank += 1
            ranks.append(rank)
    return torch.Tensor(ranks)


def specific_ranking(h_or_t, entities_num, batch_size, samples, to_be_filtered, model):
    mrr = 0
    mr = 0
    hits1 = 0
    hits3 = 0
    hits10 = 0
    for i in range(batch_size):
        if h_or_t == 't':
            head, rel, tail, ts = samples[i]
            rank_list = [(head, rel, j, ts) for j in range(entities_num)]
            rank_list = list(set(rank_list) - to_be_filtered)
            rank_list = [head, rel, tail, ts] + rank_list
            rank_list = torch.LongTensor(rank_list)
            scores = model.forward(rank_list)
            ranks = (scores > scores[0]).sum() + 1
        else:
            head, rel, tail, ts = samples[i]
            rank_list = [(j, rel, tail, ts) for j in range(entities_num)]
            rank_list = list(set(rank_list) - to_be_filtered)
            rank_list = [head, rel, tail, ts] + rank_list
            rank_list = torch.LongTensor(rank_list)
            scores = model.forward(rank_list)
            ranks = (scores > scores[0]).sum() + 1
        mrr += 1.0/ranks
        mr += ranks
        if ranks == 1:
            hits1 += 1
        if ranks <= 3:
            hits3 += 1
        if ranks <= 10:
            hits10 += 1

    return mr/batch_size, mrr/batch_size , hits1, hits3, hits10




def metric(h_or_t, entities_num, scores, to_be_filtered, test_data, batch_size, hits = [1, 3, 10]):
    with torch.no_grad():
        head = test_data[:, 0]
        rel = test_data[:, 1]
        tail = test_data[:, 2]
        time = test_data[:, 3]
        #scores = scores[0] + scores[1] + scores[2]
        ranks = ranking(h_or_t, entities_num, scores, batch_size, head, rel, tail, time, to_be_filtered)
        mrr = torch.mean(1.0 / ranks)
        mr = torch.mean(ranks)
        hits1 = torch.sum(ranks <= hits[0])
        hits3 = torch.sum(ranks <= hits[1])
        hits10 = torch.sum(ranks <= hits[2])

        return mr.item(), mrr.item(), hits1.item(), hits3.item(), hits10.item()



def neg_sample(samples, neg_size, h_or_t, entities_num):
    if h_or_t == 't':
        batch_neg_samples = []
        for i in range(samples.shape[0]):
            samples_masked = []
            length = 0
            s = samples[i]
            while length < neg_size:
                negative_tails = np.random.randint(entities_num, size=neg_size)
                mask = np.in1d(negative_tails, s[2], invert=True)
                negative_tails = negative_tails[mask]
                samples_masked.append(negative_tails)
                length += negative_tails.shape[0]
            all_neg_samples = np.concatenate(samples_masked)
            negative_samples = all_neg_samples[:neg_size]
            batch_neg_samples.append(negative_samples)
        batch_neg_samples = np.concatenate(batch_neg_samples)
        batch_neg_samples = batch_neg_samples.reshape(samples.shape[0], neg_size)
    else:
        batch_neg_samples = []
        for i in range(samples.shape[0]):
            samples_masked = []
            length = 0
            s = samples[i]
            while length < neg_size:
                negative_tails = np.random.randint(entities_num, size=neg_size)
                mask = np.in1d(negative_tails, s[0], invert=True)
                negative_tails = negative_tails[mask]
                samples_masked.append(negative_tails)
                length += negative_tails.shape[0]
            all_neg_samples = np.concatenate(samples_masked)
            negative_samples = all_neg_samples[:neg_size]
            batch_neg_samples.append(negative_samples)
        batch_neg_samples = np.concatenate(batch_neg_samples)
        batch_neg_samples = batch_neg_samples.reshape(samples.shape[0], neg_size)

    return batch_neg_samples

