from model.layers import CATS_Attention
from data.utils import InputSentenceCATSDatasetBuilder
import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.random import seed
seed(42)
import tensorflow as tf
import pickle
from sklearn.metrics import roc_auc_score
import argparse
import math
import time

class CATSSentenceModel(nn.Module):
    def __init__(self, emb_size):
        super(CATSSentenceModel, self).__init__()
        self.cats = CATS_Attention(emb_size)

    def forward(self, X):
        self.pair_scores = self.cats(X)
        return self.pair_scores

def run_model(qry_attn_file_train, qry_attn_file_test, train_pids_file, test_pids_file, train_pvecs_file,
              test_pvecs_file, train_qids_file, test_qids_file, train_qvecs_file, test_qvecs_file, use_cache,
              lrate, batch, epochs, save):
    if not use_cache:
        qry_attn_tr = []
        qry_attn_ts = []
        with open(qry_attn_file_train, 'r') as trf:
            f = True
            for l in trf:
                if f:
                    f = False
                    continue
                qry_attn_tr.append(l.split('\t'))
        with open(qry_attn_file_test, 'r') as tsf:
            f = True
            for l in tsf:
                if f:
                    f = False
                    continue
                qry_attn_ts.append(l.split('\t'))
        train_pids = np.load(train_pids_file)
        train_pvecs = np.load(train_pvecs_file)
        train_qids = np.load(train_qids_file)
        train_qvecs = np.load(train_qvecs_file)
        if train_pids_file == test_pids_file:
            test_pids = train_pids
            test_pvecs = train_pvecs
        else:
            test_pids = np.load(test_pids_file)
            test_pvecs = np.load(test_pvecs_file)
        test_qids = np.load(test_qids_file)
        test_qvecs = np.load(test_qvecs_file)

        print('Building train data')
        train_data_builder = InputSentenceCATSDatasetBuilder(qry_attn_tr, train_pids, train_pvecs, train_qids, train_qvecs)
        X_train, y_train = train_data_builder.build_input_data()
        print('Building test data')
        test_data_builder = InputSentenceCATSDatasetBuilder(qry_attn_ts, test_pids, test_pvecs, test_qids, test_qvecs)
        X_test, y_test = test_data_builder.build_input_data()

        val_split_ratio = 0.1
        val_sample_size = int(X_train.shape[0] * val_split_ratio)

        X_val = X_train[:val_sample_size]
        y_val = y_train[:val_sample_size]
        X_train = X_train[val_sample_size:]
        y_train = y_train[val_sample_size:]

        np.save('cache/X_train.npy', X_train)
        np.save('cache/y_train.npy', y_train)
        np.save('cache/X_val.npy', X_val)
        np.save('cache/y_val.npy', y_val)
        np.save('cache/X_test.npy', X_test)
        np.save('cache/y_test.npy', y_test)
    else:
        X_train = torch.tensor(np.load('cache/X_train.npy'))
        y_train = torch.tensor(np.load('cache/y_train.npy'))
        X_val = torch.tensor(np.load('cache/X_val.npy'))
        y_val = torch.tensor(np.load('cache/y_val.npy'))
        X_test = torch.tensor(np.load('cache/X_test.npy'))
        y_test = torch.tensor(np.load('cache/y_test.npy'))

    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device('cuda:0'))
    else:
        torch.cuda.set_device(torch.device('cpu'))

def main():
    parser = argparse.ArgumentParser(description='Run CATS sentwise model')
    parser.add_argument('-dd', '--data_dir', default="/home/sk1105/sumanta/CATS_data/")
    parser.add_argument('-qtr', '--qry_attn_train', default="half-y1train-qry-attn-first10k.tsv")
    parser.add_argument('-qt', '--qry_attn_test', default="by1test-qry-attn-bal-allpos-for-eval.tsv")
    parser.add_argument('-trp', '--train_pids', default="half-y1train-qry-attn-paraids-sentwise.npy")
    parser.add_argument('-tp', '--test_pids', default="by1test-allpos-for-eval-pids-sentwise.npy")
    parser.add_argument('-trv', '--train_pvecs', default="half-y1train-qry-attn-paravecs-sentwise.npy")
    parser.add_argument('-tv', '--test_pvecs', default="by1test-allpos-for-eval-paravecs-sentwise.npy")
    parser.add_argument('-trq', '--train_qids', default="half-y1train-context-qids.npy")
    parser.add_argument('-tq', '--test_qids', default="by1test-context-qids.npy")
    parser.add_argument('-trqv', '--train_qvecs', default="half-y1train-context-qvecs.npy")
    parser.add_argument('-tqv', '--test_qvecs', default="by1test-context-qvecs.npy")
    parser.add_argument('-lr', '--lrate', type=float, default=0.0001)
    parser.add_argument('-bt', '--batch', type=int, default=32)
    parser.add_argument('-ep', '--epochs', type=int, default=5)
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    dat = args.data_dir

    run_model(dat+args.qry_attn_train, dat+args.qry_attn_test, dat+args.train_pids, dat+args.test_pids, dat+args.train_pvecs,
              dat+args.test_pvecs, dat+args.train_qids, dat+args.test_qids, dat+args.train_qvecs, dat+args.test_qvecs,
              args.cache, args.lrate, args.batch, args.epochs, args.save)


if __name__ == '__main__':
    main()