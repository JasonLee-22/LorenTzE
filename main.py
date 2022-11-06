# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import argparse

args = argparse.ArgumentParser()

args.add_argument('--batch_size', type=int, default=1024)
args.add_argument('--epochs', type=int, default=1000)
args.add_argument('--valid_epochs', type=int, default=5)
args.add_argument('--lr', type=float, default=0.0001)
args.add_argument('--h_or_t', type=str, default='t')
args.add_argument('--dropout', type=float, default=0.4)
args.add_argument('--dataset', type=str, default='ICEWS14')
args.add_argument('--name', type=str, default=' ')
args.add_argument('--negative_size', type = int, default=128)
args = args.parse_args()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
