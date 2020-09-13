from model.layers import CATS, CATS_Scaled
from data.utils import InputCATSDatasetBuilder
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tensorflow as tf
import pickle
from sklearn.metrics import roc_auc_score

class SimilarityClusteringModel(nn.Module):
    def __init__(self, emb_size, m):
        super(SimilarityClusteringModel, self).__init__()
        self.cats = CATS(emb_size)
        self.m = m

    '''
    X is a 3D tensor of shape (n X mC2 X 3*v) where n = num of samples, m = num of paras for each query
    and v = emb vec length
    '''
    def forward(self, X):
        self.pair_score_matrix = self.cats(X)
        tf.sim_matrix = tf.map_fn(self.arrange_simscore_in_sim_matrix, self.pair_score_matrix)
        '''
        TODO
        '''

    def arrange_simscore_in_sim_matrix(self, s):
        assert self.m * (self.m - 1) / 2 == len(s)
        sim_matrix = np.zeros((self.m, self.m))
        n = 0
        for i in range(self.m):
            for j in range(self.m):
                if i==j:
                    sim_matrix[i][j] = 1.0
                elif i<j:
                    sim_matrix[i][j] = s[n]
                    n += 1
                else:
                    sim_matrix[i][j] = sim_matrix[j][i]
        return sim_matrix

class CATSSimilarityModel(nn.Module):
    def __init__(self, emb_size):
        super(CATSSimilarityModel, self).__init__()
        self.cats = CATS(emb_size)

    def forward(self, X):
        self.pair_scores = self.cats(X)
        return self.pair_scores

def run_model(qry_attn_file_train, qry_attn_file_test, train_pids_file, test_pids_file, train_pvecs_file,
              test_pvecs_file, train_qids_file, test_qids_file, train_qvecs_file, test_qvecs_file):
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

    train_data_builder = InputCATSDatasetBuilder(qry_attn_tr, train_pids, train_pvecs, train_qids, train_qvecs)
    X_train, y_train = train_data_builder.build_input_data()
    test_data_builder = InputCATSDatasetBuilder(qry_attn_ts, test_pids, test_pvecs, test_qids, test_qvecs)
    X_test, y_test = test_data_builder.build_input_data()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    y_cos = cos(X_test[:,768:768*2], X_test[:,768*2:])
    print(y_cos)
    cos_auc = roc_auc_score(y_test, y_cos)
    print('Test cosine auc: '+str(cos_auc))

    val_split_ratio = 0.1
    val_sample_size = int(X_train.shape[0] * val_split_ratio)

    X_val = X_train[:val_sample_size]
    y_val = y_train[:val_sample_size]
    X_train = X_train[val_sample_size:]
    y_train = y_train[val_sample_size:]

    m = CATSSimilarityModel(768)
    opt = optim.Adam(m.parameters(), lr=0.0001)
    mseloss = nn.MSELoss()
    for i in range(100):
        m.train()
        opt.zero_grad()
        ypred = m(X_train)
        loss = mseloss(ypred, y_train)
        ypred = ypred.detach().numpy()
        auc = roc_auc_score(y_train, ypred)
        loss.backward()
        opt.step()
        if i % 10 == 0:
            m.eval()
            ypred_val = m(X_val)
            val_loss = mseloss(ypred_val, y_val)
            ypred_val = ypred_val.detach().numpy()
            val_auc = roc_auc_score(y_val, ypred_val)
            print('Train loss: '+str(loss)+', Train auc: '+str(auc)+', Val loss: '+str(val_loss)+', Val auc: '+str(val_auc))
    m.eval()
    ypred_test = m(X_test)
    test_loss = mseloss(ypred_test, y_test)
    ypred_test = ypred_test.detach().numpy()
    test_auc = roc_auc_score(y_test, ypred_test)
    print('Test loss: '+str(test_loss)+', Test auc: '+str(test_auc))



def main():
    data_dir = "/home/sk1105/sumanta/CATS_data/"
    qry_attn_file_train = data_dir + "half-y1train-qry-attn.tsv"
    qry_attn_file_test = data_dir + "by1test-qry-attn.tsv"
    train_pids_file = data_dir + "half-y1train-qry-attn-paraids.npy"
    test_pids_file = data_dir + "by1train_by1test_by2test_pids.npy"
    train_pvecs_file = data_dir + "half-y1train-qry-attn-paravecs.npy"
    test_pvecs_file = data_dir + "by1train_by1test_by2test_vecs.npy"
    train_qids_file = data_dir + "half-y1train-context-qids.npy"
    test_qids_file = data_dir + "by1test-context-qids.npy"
    train_qvecs_file = data_dir + "half-y1train-context-qvecs.npy"
    test_qvecs_file = data_dir + "by1test-context-qvecs.npy"

    run_model(qry_attn_file_train, qry_attn_file_test, train_pids_file, test_pids_file, train_pvecs_file,
              test_pvecs_file, train_qids_file, test_qids_file, train_qvecs_file, test_qvecs_file)


if __name__ == '__main__':
    main()