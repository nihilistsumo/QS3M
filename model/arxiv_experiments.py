import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import random
import time
import math
import os
import argparse
from model.models import CATSSimilarityModel
random.seed(42)
torch.manual_seed(42)
from numpy.random import seed
seed(42)


def arxiv_experiment(arxiv_vecs, arxiv_qlabel, query_map, sbert_model_name, select_queries, lrate, epochs, batch,
                     eval_num, save):
    bert_embed_model = models.Transformer(sbert_model_name, max_seq_length=512)
    pooling_model = models.Pooling(bert_embed_model.get_word_embedding_dimension(), pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False, pooling_mode_mean_tokens=True)
    model = SentenceTransformer(modules=[bert_embed_model, pooling_model])
    queries, texts = [], []
    for q in query_map.keys():
        queries.append(q)
        texts.append(query_map[q])
    vecs = model.encode(texts)
    query_vecs = {}
    for i in range(len(queries)):
        query_vecs[queries[i]] = vecs[i]
    abs_vecs = np.load(arxiv_vecs, allow_pickle=True)[()]['data']
    abs_qlabels = np.load(arxiv_qlabel, allow_pickle=True)[()]['data']
    bal_pairs, labels = [], []
    for q in select_queries:
        dat = abs_qlabels[q]
        for k in dat.keys():
            curr_docs = dat[k]
            for i in range(len(curr_docs)-1):
                for j in range(1, len(curr_docs)):
                    bal_pairs.append((q, curr_docs[i], curr_docs[j]))
                    labels.append(1)
                    neg_k = random.sample(list(dat.keys()), 1)[0]
                    while neg_k == k:
                        neg_k = random.sample(list(dat.keys()), 1)[0]
                    bal_pairs.append((q, curr_docs[i], random.sample(dat[k], 1)[0]))
                    labels.append(0)
    xdata = torch.zeros((len(bal_pairs), 3*bert_embed_model.get_word_embedding_dimension()))
    ydata = torch.tensor(labels, dtype=torch.float32)
    for i in range(len(bal_pairs)):
        xdata[i] = torch.tensor(np.hstack((query_vecs[bal_pairs[i][0]], abs_vecs[bal_pairs[i][1]], abs_vecs[bal_pairs[i][2]])))
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('CUDA available, using CUDA')
    else:
        device = torch.device('cpu')
        print('CUDA not available, using cpu')
    fold = 1
    for train_index, test_index in skf.split(xdata, ydata):
        print('Fold %d' % fold)
        X_train, X_test = xdata[train_index], xdata[test_index]
        y_train, y_test = ydata[train_index], ydata[test_index]

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        y_cos = cos(X_test[:, 768:768 * 2], X_test[:, 768 * 2:])
        cos_auc = roc_auc_score(y_test.detach().cpu().numpy(), y_cos.detach().cpu().numpy())
        print('Test data Baseline cosine auc: %.5f', cos_auc)
        y_euclid = torch.sqrt(torch.sum((X_test[:, 768:768 * 2] - X_test[:, 768 * 2:]) ** 2, 1)).detach().cpu().numpy()
        y_euclid = 1 - (y_euclid - np.min(y_euclid)) / (np.max(y_euclid) - np.min(y_euclid))
        euclid_auc = roc_auc_score(y_test.detach().cpu().numpy(), y_euclid)
        print('Test data Baseline euclidean auc: %.5f', euclid_auc)

        train_samples = X_train.shape[0]
        m = CATSSimilarityModel(768, 'cats').to(device)
        m.cats.to(device)
        opt = optim.Adam(m.parameters(), lr=lrate)
        mseloss = nn.MSELoss()
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        print('Starting training...')
        for i in range(epochs):
            print('\nEpoch ' + str(i + 1))
            for b in range(math.ceil(train_samples // batch)):
                m.train()
                opt.zero_grad()
                curr_x = X_train[b * batch:b * batch + batch].to(device)
                ypred = m(curr_x)
                y_train_curr = y_train[b * batch:b * batch + batch].to(device)
                loss = mseloss(ypred, y_train_curr)
                auc = roc_auc_score(y_train_curr.detach().cpu().numpy(), ypred.detach().cpu().numpy())
                loss.backward()
                opt.step()
                if b % eval_num == 0:
                    m.eval()
                    ypred_test = m(X_test)
                    test_loss = mseloss(ypred_test, y_test)
                    val_auc = roc_auc_score(y_test.detach().cpu().numpy(), ypred_test.detach().cpu().numpy())
                    print(
                        '\rTrain loss: %.5f, Train auc: %.5f, Test loss: %.5f, Test auc: %.5f' %
                        (loss.item(), auc, test_loss.item(), val_auc), end='')
        m.eval()
        ypred_test = m(X_test)
        test_loss = mseloss(ypred_test, y_test)
        test_auc = roc_auc_score(y_test.detach().cpu().numpy(), ypred_test.detach().cpu().numpy())
        print('\n\nTest loss: %.5f, Test auc: %.5f' % (test_loss.item(), test_auc))

        if save:
            if not os.path.isdir('saved_models'):
                os.makedirs('saved_models')
            torch.save(m.state_dict(), 'saved_models/fold-' + str(fold) + time.strftime('%b-%d-%Y_%H%M', time.localtime()) + '.model')
        fold += 1


def main():
    parser = argparse.ArgumentParser(description='Run CATS model')
    parser.add_argument('-av', '--arxiv_vecs', default='/home/sk1105/sumanta/arxiv_data_for_cats/arxiv_vecs_for_cats.npy')
    parser.add_argument('-ql', '--arxiv_qlabels', default='/home/sk1105/sumanta/arxiv_data_for_cats/arxiv_qlabels_for_cats.npy')
    parser.add_argument('-lr', '--lrate', type=float, default=0.00001)
    parser.add_argument('-ep', '--epochs', type=int, default=3)
    parser.add_argument('-bt', '--batch', type=int, default=32)
    parser.add_argument('-en', '--eval_num', type=int, default=100)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    selected_queries = ['eess', 'astro-ph', 'cond-mat', 'nlin', 'q-bio', 'q-fin', 'stat']
    query_map = {'cs': 'Computer Science',
             'econ': 'Economics',
             'eess': 'Electrical Engineering and Systems',
             'math': 'Mathematics',
             'astro-ph': 'Astrophysics',
             'cond-mat': 'Condensed Matter',
             'nlin': 'Nonlinear',
             'physics': 'Physics',
             'q-bio': 'Biology',
             'q-fin': 'Finance',
             'stat': 'Statistics'}
    arxiv_experiment(args.arxiv_vecs, args.arxiv_qlabels, query_map, 'bert-base-uncased', selected_queries, args.lrate,
                     args.epochs, args.batch, args.eval_num, args.save)

if __name__ == '__main__':
    main()
