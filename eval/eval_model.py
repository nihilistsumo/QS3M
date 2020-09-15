from model.layers import CATS, CATS_Scaled, CATS_QueryScaler, CATS_manhattan
from data.utils import InputCATSDatasetBuilder
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
from sklearn.metrics.pairwise import euclidean_distances
import argparse
import math
import time

def eval_cluster(model_path, model_type, qry_attn_file_test, test_pids_file, test_pvecs_file, test_qids_file,
                 test_qvecs_file, use_cache):
    if model_type == 'triam':
        model = CATS(768)
    elif model_type == 'qscale':
        model = CATS_QueryScaler(768)
    else:
        print('Wrong model type')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if not use_cache:
        qry_attn_ts = []
        with open(qry_attn_file_test, 'r') as tsf:
            f = True
            for l in tsf:
                if f:
                    f = False
                    continue
                qry_attn_ts.append(l.split('\t'))
        test_pids = np.load(test_pids_file)
        test_pvecs = np.load(test_pvecs_file)
        test_qids = np.load(test_qids_file)
        test_qvecs = np.load(test_qvecs_file)

        test_data_builder = InputCATSDatasetBuilder(qry_attn_ts, test_pids, test_pvecs, test_qids, test_qvecs)
        X_test, y_test = test_data_builder.build_input_data()

        np.save('cache/X_test.npy', X_test)
        np.save('cache/y_test.npy', y_test)
    else:
        X_test = torch.tensor(np.load('cache/X_test.npy'))
        y_test = torch.tensor(np.load('cache/y_test.npy'))

    model.cpu()
    ypred_test = model(X_test)
    mseloss = nn.MSELoss()
    test_loss = mseloss(ypred_test, y_test)
    test_auc = roc_auc_score(y_test.detach().cpu().numpy(), ypred_test.detach().cpu().numpy())
    print('\n\nTest loss: %.5f, Test auc: %.5f' % (test_loss.item(), test_auc))

def main():
    parser = argparse.ArgumentParser(description='Run CATS model')
    parser.add_argument('-dd', '--data_dir', default="/home/sk1105/sumanta/CATS_data/")
    parser.add_argument('-qt', '--qry_attn_test', default="by1test-qry-attn.tsv")
    parser.add_argument('-tp', '--test_pids', default="by1test-allpos-for-eval-pids.npy")
    parser.add_argument('-tv', '--test_pvecs', default="by1test-allpos-for-eval-paravecs.npy")
    parser.add_argument('-tq', '--test_qids', default="by1test-context-qids.npy")
    parser.add_argument('-tqv', '--test_qvecs', default="by1test-context-qvecs.npy")
    parser.add_argument('-mt', '--model_type', default="triam")
    parser.add_argument('-mp', '--model_path', default="/home/sk1105/CATS/saved_models/cats_2triamese_layer_b32_l0.00001_i5.model")
    parser.add_argument('--cache', action='store_true')

    args = parser.parse_args()
    dat = args.data_dir

    eval_cluster(args.model_path, args.model_type, args.qry_attn_test, args.test_pids, args.test_pvecs, args.test_qids,
                 args.test_qvecs, args.cache)

if __name__ == '__main__':
    main()