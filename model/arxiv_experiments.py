import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
import random
import time
import math
import os
import argparse
from tqdm import tqdm
from model.models import CATSSimilarityModel
random.seed(42)
torch.manual_seed(42)
from numpy.random import seed
seed(42)


class CATS_arxiv(nn.Module): # CATS
    def __init__(self, emb_size):
        super(CATS_arxiv, self).__init__()
        self.emb_size = emb_size
        self.LL1 = nn.Linear(emb_size, 5 * emb_size)
        self.LL2 = nn.Linear(5 * emb_size, emb_size)
        self.LL3 = nn.Linear(5 * emb_size, 1)

    def forward(self, X):
        '''

        :param X: The input tensor is of shape (mC2 X 3*vec size) where m = num of paras for each query
        :return s: Pairwise CATS scores of shape (mC2 X 1)
        '''
        self.Xq = X[:, :self.emb_size]
        self.Xp1 = X[:, self.emb_size:2 * self.emb_size]
        self.Xp2 = X[:, 2 * self.emb_size:]
        self.z1 = torch.abs(self.Xp1 - self.Xq)
        self.z2 = torch.abs(self.Xp2 - self.Xq)
        self.zdiff = torch.abs(self.Xp1 - self.Xp2)
        self.zp1 = torch.relu(self.LL2(torch.relu(self.LL1(self.Xp1))))
        self.zp2 = torch.relu(self.LL2(torch.relu(self.LL1(self.Xp2))))
        self.zql = torch.relu(self.LL2(torch.relu(self.LL1(self.Xq))))
        self.zd = torch.abs(self.zp1 - self.zp2)
        self.zdqp1 = torch.abs(self.zp1 - self.zql)
        self.zdqp2 = torch.abs(self.zp2 - self.zql)
        self.z = torch.cat((self.zp1, self.zp2, self.zd, self.zdqp1, self.zdqp2), dim=1)
        o = torch.relu(self.LL3(self.z))
        o = o.reshape(-1)
        return o

    def num_flat_features(self, X):
        size = X.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, X_test):
        y_pred = self.forward(X_test)
        return y_pred

class CATSSimilarityModel_arxiv(nn.Module):
    def __init__(self, emb_size):
        super(CATSSimilarityModel_arxiv, self).__init__()
        self.cats = CATS_arxiv(emb_size)

    def forward(self, X):
        self.pair_scores = self.cats(X)
        return self.pair_scores


def arxiv_experiment(arxiv_qlabel, query_map, sbert_model_name, select_queries, lrate, epochs, batch,
                     eval_num, save, arxiv_vecs=None, arxiv_docs=None):
    if '/' in sbert_model_name:
        model = SentenceTransformer(sbert_model_name)
    else:
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
    if arxiv_vecs is not None:
        abs_vecs = np.load(arxiv_vecs, allow_pickle=True)[()]['data']
    else:
        abstracts = np.load(arxiv_docs, allow_pickle=True)[()]['data']
        abstract_ids, abstract_texts = [], []
        for a in abstracts.keys():
            abstract_ids.append(a)
            abstract_texts.append(abstracts[a])
        abs_vecs = {}
        print('Going to embed paras')
        embeds = model.encode(abstract_texts, show_progress_bar=True)
        for i in range(len(abstract_ids)):
            abs_vecs[abstract_ids[i]] = embeds[i]
    abs_qlabels = np.load(arxiv_qlabel, allow_pickle=True)[()]['data']
    qdocs, labels = [], []
    for q in abs_qlabels.keys():
        for k in abs_qlabels[q].keys():
            for d in abs_qlabels[q][k]:
                qdocs.append({'a': d, 'q': q, 'cat': k})
                labels.append(q+':'+k)
    skf = StratifiedKFold(n_splits=2)
    for train_index, test_index in skf.split(qdocs, labels):
        train_qdocs = [qdocs[i] for i in train_index]
        test_qdocs = [qdocs[i] for i in test_index]
        abs_qlabels_train, abs_qlabels_test = {}, {}
        for d in train_qdocs:
            q, k, d = d['q'], d['cat'], d['a']
            if q in abs_qlabels_train.keys():
                if k in abs_qlabels_train[q].keys():
                    abs_qlabels_train[q][k].append(d)
                else:
                    abs_qlabels_train[q][k] = [d]
            else:
                abs_qlabels_train[q] = {k: [d]}
        for d in test_qdocs:
            q, k, d = d['q'], d['cat'], d['a']
            if q in abs_qlabels_test.keys():
                if k in abs_qlabels_test[q].keys():
                    abs_qlabels_test[q][k].append(d)
                else:
                    abs_qlabels_test[q][k] = [d]
            else:
                abs_qlabels_test[q] = {k: [d]}


        bal_pairs, labels = [], []
        for q in select_queries:
            dat = abs_qlabels_train[q]
            for k in dat.keys():
                curr_docs = dat[k]
                for i in range(len(curr_docs)-1):
                    for j in range(1, len(curr_docs)):
                        bal_pairs.append((q, curr_docs[i], curr_docs[j]))
                        labels.append(1)
                        neg_k = random.sample(list(dat.keys()), 1)[0]
                        while neg_k == k:
                            neg_k = random.sample(list(dat.keys()), 1)[0]
                        bal_pairs.append((q, curr_docs[i], random.sample(dat[neg_k], 1)[0]))
                        labels.append(0)
        emb_dim = model.get_sentence_embedding_dimension()
        X_train = torch.zeros((len(bal_pairs), 3*emb_dim))
        y_train = torch.tensor(labels, dtype=torch.float32)
        for i in range(len(bal_pairs)):
            X_train[i] = torch.tensor(np.hstack((query_vecs[bal_pairs[i][0]], abs_vecs[bal_pairs[i][1]], abs_vecs[bal_pairs[i][2]])))

        bal_pairs_test, labels_test = [], []
        for q in select_queries:
            dat = abs_qlabels_test[q]
            for k in dat.keys():
                curr_docs = dat[k]
                for i in range(len(curr_docs) - 1):
                    for j in range(1, len(curr_docs)):
                        bal_pairs_test.append((q, curr_docs[i], curr_docs[j]))
                        labels_test.append(1)
                        neg_k = random.sample(list(dat.keys()), 1)[0]
                        while neg_k == k:
                            neg_k = random.sample(list(dat.keys()), 1)[0]
                        bal_pairs_test.append((q, curr_docs[i], random.sample(dat[neg_k], 1)[0]))
                        labels_test.append(0)
        X_test = torch.zeros((len(bal_pairs_test), 3 * emb_dim))
        y_test = torch.tensor(labels_test, dtype=torch.float32)
        for i in range(len(bal_pairs_test)):
            X_test[i] = torch.tensor(np.hstack((query_vecs[bal_pairs_test[i][0]], abs_vecs[bal_pairs_test[i][1]], abs_vecs[bal_pairs_test[i][2]])))

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            print('CUDA available, using CUDA')
        else:
            device = torch.device('cpu')
            print('CUDA not available, using cpu')
        fold = 1

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        y_cos = cos(X_test[:, emb_dim:emb_dim * 2], X_test[:, emb_dim * 2:])
        cos_auc = roc_auc_score(y_test.detach().cpu().numpy(), y_cos.detach().cpu().numpy())
        print('Test data Baseline cosine auc: %.5f', cos_auc)
        y_euclid = torch.sqrt(torch.sum((X_test[:, emb_dim:emb_dim * 2] - X_test[:, emb_dim * 2:]) ** 2, 1)).detach().cpu().numpy()
        y_euclid = 1 - (y_euclid - np.min(y_euclid)) / (np.max(y_euclid) - np.min(y_euclid))
        euclid_auc = roc_auc_score(y_test.detach().cpu().numpy(), y_euclid)
        print('Test data Baseline euclidean auc: %.5f', euclid_auc)
        print('Test data clustering eval')
        rands = []
        for q in select_queries:
            dat = abs_qlabels_test[q]
            docs, l = [], []
            for k in dat.keys():
                docs += dat[k]
                l += [k] * len(dat[k])
            vecs = np.zeros((len(docs), emb_dim))
            for i in range(len(docs)):
                vecs[i] = abs_vecs[docs[i]]
            k = len(set(l))
            score_matrix = cosine_distances(vecs, vecs)
            cl = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
            cl_labels = cl.fit_predict(score_matrix)
            base_rand = adjusted_rand_score(l, cl_labels)
            rands.append(base_rand)
            print(q+' ARI: %.5f' % base_rand)
        rands = np.array(rands)
        print('Test data mean ARI: %.5f, stderr: %.5f' % (np.mean(rands), np.std(rands)))
        print('\n\n')

        train_samples = X_train.shape[0]
        test_samples = X_test.shape[0]
        m = CATSSimilarityModel(emb_dim, 'cats').to(device)
        m.cats.to(device)
        opt = optim.Adam(m.parameters(), lr=lrate)
        mseloss = nn.MSELoss()
        X_test = X_test.to(device)
        y_test = y_test.to(device)
        print('Starting training...')
        for i in range(epochs):
            print('\nEpoch ' + str(i + 1))
            for b in tqdm(range(math.ceil(train_samples // batch))):
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
                    test_loss = 0
                    test_auc = 0
                    n = 0
                    for bt in range(math.ceil(test_samples // batch)):
                        curr_x_test = X_test[bt * batch:bt * batch + batch].to(device)
                        y_test_curr = y_test[bt * batch:bt * batch + batch].to(device)
                        ypred_test = m(curr_x_test)
                        test_loss += mseloss(ypred_test, y_test_curr).item()
                        test_auc += roc_auc_score(y_test_curr.detach().cpu().numpy(), ypred_test.detach().cpu().numpy())
                        n += 1
                    print(
                        '\rTrain loss: %.5f, Train auc: %.5f, Test loss: %.5f, Test auc: %.5f' %
                        (loss.item(), auc, test_loss/n, test_auc/n), end='')
        m.eval()
        test_loss = 0
        test_auc = 0
        n = 0
        for bt in range(math.ceil(test_samples // batch)):
            curr_x_test = X_test[bt * batch:bt * batch + batch].to(device)
            y_test_curr = y_test[bt * batch:bt * batch + batch].to(device)
            ypred_test = m(curr_x_test)
            test_loss += mseloss(ypred_test, y_test_curr).item()
            test_auc += roc_auc_score(y_test_curr.detach().cpu().numpy(), ypred_test.detach().cpu().numpy())
            n += 1
        print(
            '\rTrain loss: %.5f, Train auc: %.5f, Test loss: %.5f, Test auc: %.5f' %
            (loss.item(), auc, test_loss / n, test_auc / n), end='')
        print('\n\nTest loss: %.5f, Test auc: %.5f' % (test_loss/n, test_auc/n))
        print('Test cluster eval')
        rands = []
        for q in select_queries:
            dat = abs_qlabels_test[q]
            docs, l = [], []
            for k in dat.keys():
                docs += dat[k]
                l += [k] * len(dat[k])
            vecs = np.zeros((len(docs), emb_dim))
            for i in range(len(docs)):
                vecs[i] = abs_vecs[docs[i]]
            k = len(set(l))
            score_matrix = np.zeros((len(docs), len(docs)))
            for i in range(len(docs)):
                rdata = torch.zeros((len(docs), 3*emb_dim))
                for j in range(len(docs)):
                    rdata[j] = torch.tensor(np.hstack((query_vecs[q], abs_vecs[docs[i]], abs_vecs[docs[j]])))
                rdata = rdata.to(device)
                rdata_scores = m(rdata).detach().cpu().numpy()
                score_matrix[i] = rdata_scores
            score_matrix = (score_matrix - np.min(score_matrix))/(np.max(score_matrix) - np.min(score_matrix))
            score_matrix = 1 - score_matrix
            cl = AgglomerativeClustering(n_clusters=k, affinity='precomputed', linkage='average')
            cl_labels = cl.fit_predict(score_matrix)
            base_rand = adjusted_rand_score(l, cl_labels)
            rands.append(base_rand)
            print(q+' ARI: %.5f' % base_rand)
        rands = np.array(rands)
        print('Test data mean ARI: %.5f, stderr: %.5f' % (np.mean(rands), np.std(rands)))
        print('\n\n')



        if save:
            if not os.path.isdir('saved_models'):
                os.makedirs('saved_models')
            torch.save(m.state_dict(), 'saved_models/fold-' + str(fold) + time.strftime('%b-%d-%Y_%H%M', time.localtime()) + '.model')
        fold += 1


def main():
    parser = argparse.ArgumentParser(description='Run CATS model')
    parser.add_argument('-ad', '--arxiv_docs', default=None)
    parser.add_argument('-av', '--arxiv_vecs', default=None)
    parser.add_argument('-ql', '--arxiv_qlabels', default='/home/sk1105/sumanta/arxiv_data_for_cats/arxiv_qlabels_for_cats.npy')
    parser.add_argument('-mn', '--model_name', default='bert-base-uncased')
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
    arxiv_experiment(args.arxiv_qlabels, query_map, args.model_name, selected_queries, args.lrate,
                     args.epochs, args.batch, args.eval_num, args.save, args.arxiv_vecs, args.arxiv_docs)

if __name__ == '__main__':
    main()
