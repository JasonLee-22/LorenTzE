import argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='ICEWS14')

args = args.parse_args()

dataset = args.dataset

if dataset == 'ICEWS14':
    f_train = open('./data/ICEWS14')
elif dataset == 'GDELT':
    f_train = open('./data/GDELT')

