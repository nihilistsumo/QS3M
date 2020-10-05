import numpy as np
from itertools import combinations
from hashlib import sha1
from sentence_transformers import SentenceTransformer
import torch
torch.manual_seed(42)

class InputClusterDatasetBuilder:
    '''
    query_para_data: {query ID:{'paras':[], 'cluster_labels':[]}, .... }
    Each query should have equal num of paras and cluster labels, otherwise it will fail assetion
    Output from build_input_data -> X: 3D tensor of shape n X mC2 X 3*v where n = num of samples,
    m = num of paras for each query, v = emb vec length
    y: 2D tensor of shape n X mC2 containing similarity labels
    '''
    def __init__(self, query_para_data, paraids_npy, paravecs_npy, query_emb_vecs):
        paranums = [len(query_para_data[q]['paras']) for q in query_para_data.keys()]
        labelnums = [len(query_para_data[q]['cluster_labels']) for q in query_para_data.keys()]
        assert len(set(paranums)) == 1
        assert len(set(labelnums)) == 1
        paralist = []
        for q, data in query_para_data.items():
            paralist += data['paras']
        paraids = list(paraids_npy)
        assert set(paralist).issubset(set(paraids)) is True
        self.paralist = list(set(paralist))
        self.para_vecs = {}
        for p in self.paralist:
            self.para_vecs[p] = paravecs_npy[paraids.index(p)]
        self.query_para_data = query_para_data
        self.query_vecs = query_emb_vecs
        self.m = paranums[0]

    def build_input_data(self):
        para_pair_indices = [p for p in combinations(range(self.m), 2)]
        X = []
        y = []
        for q, data in self.query_para_data.items():
            paras = data['paras']
            cluster_labels = data['cluster_labels']
            qvec = list(self.query_vecs[q]['query_vec'])
            X_row = []
            y_row = []
            for pair_index in para_pair_indices:
                X_row.append(qvec + list(self.para_vecs[paras[pair_index[0]]]) +
                             list(self.para_vecs[paras[pair_index[1]]]))
                y_row.append(1 if cluster_labels[pair_index[0]] == cluster_labels[pair_index[1]] else 0)
            X.append(X_row)
            y.append(y_row)
        X = np.array(X)
        y = np.array(y)
        print('X shape: '+str(X.shape)+', y shape: '+str(y.shape))
        return X, y

class InputCATSDatasetBuilder:
    '''
    query_attn_data: [[query ID, para1 ID, para2 ID, int label], ....]
    '''
    def __init__(self, query_attn_data, paraids_npy, paravecs_npy, queryids_npy, queryvecs_npy):
        paralist = []
        querylist = []
        for data in query_attn_data:
            querylist.append(data[0])
            paralist.append(data[1])
            paralist.append(data[2])
        self.paraids = list(paraids_npy)
        self.paravecs_npy = paravecs_npy
        assert set(paralist).issubset(set(self.paraids)) is True
        queries = list(queryids_npy)
        self.query_attn_data = query_attn_data
        paralist = list(set(paralist))
        querylist = list(set(querylist))
        self.para_vecs = {}
        paraids_dict = {}
        for i, para in enumerate(self.paraids):
            paraids_dict[para] = i
        queries_dict = {}
        for i, q in enumerate(queries):
            queries_dict[q] = i
        #print('Going to initialize para vecs')
        for p in paralist:
            self.para_vecs[p] = paravecs_npy[paraids_dict[p]]
        self.query_vecs = {}
        missing_queries = set()
        for q in querylist:
            if q not in queries_dict.keys():
                missing_queries.add(q)
            else:
                self.query_vecs[q] = queryvecs_npy[queries_dict[q]]
        if len(missing_queries) > 0:
            print('Following '+str(len(missing_queries))+' queries are present in qry_attn file but not in context json file')
            print('The root cause is these queries are present in article.qrels without any / but in top-level, hier level'
                  'they are always present with /')
            for q in missing_queries:
                print(q)
        #print('Init done')

    def build_input_data(self, qry_attn_data=None):
        X = []
        y = []
        if qry_attn_data == None:
            qry_attn_data = self.query_attn_data
        for qid, pid1, pid2, label in qry_attn_data:
            if qid in self.query_vecs.keys():
                row = np.hstack((self.query_vecs[qid], self.para_vecs[pid1], self.para_vecs[pid2]))
                y.append(float(label))
                X.append(row)
        X = torch.tensor(X)
        y = torch.tensor(y)
        #print('X shape: ' + str(X.shape) + ', y shape: ' + str(y.shape))
        return X, y

    def build_input_data_with_pairs(self, qry_attn_data=None):
        X = []
        y = []
        pairs = []
        if qry_attn_data == None:
            qry_attn_data = self.query_attn_data
        for qid, pid1, pid2, label in qry_attn_data:
            if qid in self.query_vecs.keys():
                row = np.hstack((self.query_vecs[qid], self.para_vecs[pid1], self.para_vecs[pid2]))
                y.append(float(label))
                X.append(row)
                if pid1 < pid2:
                    pairs.append(pid1 + '_' + pid2)
                else:
                    pairs.append(pid2 + '_' + pid1)
        X = torch.tensor(X)
        y = torch.tensor(y)
        #print('X shape: ' + str(X.shape) + ', y shape: ' + str(y.shape))
        return X, y, pairs

    def build_cluster_data(self, qid, paralist):
        all_para_vec_dict = {}
        for i, para in enumerate(self.paraids):
            all_para_vec_dict[para] = self.paravecs_npy[i]
        X = []
        if qid not in self.query_vecs.keys():
            print(qid+' not present in query vecs dict')
            return None
        parapairs = []
        paralist.sort()
        for i in range(len(paralist)-1):
            for j in range(i+1, len(paralist)):
                p1 = paralist[i]
                p2 = paralist[j]
                row = np.hstack((self.query_vecs[qid], all_para_vec_dict[p1], all_para_vec_dict[p2]))
                X.append(row)
                parapairs.append(p1+'_'+p2)
        X = torch.tensor(X)
        #print(qid+' X shape: '+str(X.shape))
        return X, parapairs

class InputSentenceCATSDatasetBuilder:
    '''
        query_attn_data: [[query ID, para1 ID, para2 ID, int label], ....]
        '''

    def __init__(self, query_attn_data, paraids_npy, paravecs_npy, queryids_npy, queryvecs_npy, max_seq_len=5):
        paralist = []
        querylist = []
        for data in query_attn_data:
            querylist.append(data[0])
            paralist.append(data[1])
            paralist.append(data[2])
        paraids = list(paraids_npy)
        assert set(paralist).issubset(set([p[0] for p in paraids])) is True
        self.paraids_dict = {}
        for i in range(len(paraids)):
            self.paraids_dict[paraids[i][0]] = [int(paraids[i][1]), int(paraids[i][2])]
        self.paravecs_npy = paravecs_npy
        self.queryids = list(queryids_npy)
        self.queryvecs_npy = queryvecs_npy
        self.query_attn_data = query_attn_data
        self.max_seq_len = max_seq_len
        self.emb_len = 768

        self.query_indices = {}
        for i in range(len(self.queryids)):
            self.query_indices[self.queryids[i]] = i

    def build_input_data(self, qry_attn_dat=None):
        Xq = []
        Xp = []
        y = []
        if qry_attn_dat == None:
            qry_attn_dat = self.query_attn_data
        samples = len(qry_attn_dat)
        i = 0
        pairs = []
        for qid, pid1, pid2, label in qry_attn_dat:
            if qid in self.query_indices.keys():
                # row = np.hstack((self.query_vecs[qid], self.para_vecs[pid1], self.para_vecs[pid2]))
                qvec = self.queryvecs_npy[self.query_indices[qid]]
                p1index_dat = self.paraids_dict[pid1]
                p1mat = self.paravecs_npy[p1index_dat[0]:p1index_dat[0] + p1index_dat[1]]
                p1vec_len = p1mat.shape[0]
                if p1vec_len < self.max_seq_len:
                    valid_bits = np.array([1.0] * p1vec_len + [0.0] * (self.max_seq_len - p1vec_len)).reshape((-1,1))
                    z = np.zeros((self.max_seq_len - p1vec_len, self.emb_len))
                    p1mat = np.hstack((np.vstack((p1mat, z)), valid_bits))
                else:
                    valid_bits = np.array([1.0] * self.max_seq_len).reshape((-1,1))
                    p1mat = np.hstack((p1mat[:self.max_seq_len], valid_bits))

                p2index_dat = self.paraids_dict[pid2]
                p2mat = self.paravecs_npy[p2index_dat[0]:p2index_dat[0] + p2index_dat[1]]
                p2vec_len = p2mat.shape[0]
                if p2vec_len < self.max_seq_len:
                    valid_bits = np.array([1.0] * p2vec_len + [0.0] * (self.max_seq_len - p2vec_len)).reshape((-1,1))
                    p2mat = np.hstack(
                        (np.vstack((p2mat, np.zeros((self.max_seq_len - p2vec_len, self.emb_len)))), valid_bits))
                else:
                    valid_bits = np.array([1.0] * self.max_seq_len).reshape((-1,1))
                    p2mat = np.hstack((p2mat[:self.max_seq_len], valid_bits))
                dat_mat = np.hstack((p1mat, p2mat))

                y.append(float(label))
                Xq.append(qvec)
                Xp.append(dat_mat.transpose())
                if pid1 < pid2:
                    pairs.append(pid1+'_'+pid2)
                else:
                    pairs.append(pid2+'_'+pid1)
                i += 1
                if i % 10000 == 0:
                    print(str(i) + ' samples processed out of '+str(samples))
        Xq = np.array(Xq)
        Xp = np.array(Xp)
        #print('numpy conversion done')
        Xq = torch.as_tensor(Xq, dtype=torch.float)
        Xp = torch.as_tensor(Xp, dtype=torch.float)
        y = torch.tensor(y)
        #print('Xq shape: ' + str(Xq.shape) + ', Xp shape: ' + str(Xp.shape) + ', y shape: ' + str(y.shape))
        return Xq, Xp, y, pairs

    def build_cluster_data(self, qid, paralist):
        if qid not in self.query_indices.keys():
            print(qid + ' not present in query vecs dict')
            return None
        Xq = []
        Xp = []
        pairs = []
        for i in range(len(paralist) - 1):
            for j in range(i + 1, len(paralist)):
                pid1 = paralist[i]
                pid2 = paralist[j]

                qvec = self.queryvecs_npy[self.query_indices[qid]]
                p1index_dat = self.paraids_dict[pid1]
                p1mat = self.paravecs_npy[p1index_dat[0]:p1index_dat[0] + p1index_dat[1]]
                p1vec_len = p1mat.shape[0]
                if p1vec_len < self.max_seq_len:
                    valid_bits = np.array([1.0] * p1vec_len + [0.0] * (self.max_seq_len - p1vec_len)).reshape((-1, 1))
                    z = np.zeros((self.max_seq_len - p1vec_len, self.emb_len))
                    p1mat = np.hstack((np.vstack((p1mat, z)), valid_bits))
                else:
                    valid_bits = np.array([1.0] * self.max_seq_len).reshape((-1, 1))
                    p1mat = np.hstack((p1mat[:self.max_seq_len], valid_bits))

                p2index_dat = self.paraids_dict[pid2]
                p2mat = self.paravecs_npy[p2index_dat[0]:p2index_dat[0] + p2index_dat[1]]
                p2vec_len = p2mat.shape[0]
                if p2vec_len < self.max_seq_len:
                    valid_bits = np.array([1.0] * p2vec_len + [0.0] * (self.max_seq_len - p2vec_len)).reshape((-1, 1))
                    p2mat = np.hstack(
                        (np.vstack((p2mat, np.zeros((self.max_seq_len - p2vec_len, self.emb_len)))), valid_bits))
                else:
                    valid_bits = np.array([1.0] * self.max_seq_len).reshape((-1, 1))
                    p2mat = np.hstack((p2mat[:self.max_seq_len], valid_bits))
                dat_mat = np.hstack((p1mat, p2mat))

                Xq.append(qvec)
                Xp.append(dat_mat.transpose())
                if pid1 < pid2:
                    pairs.append(pid1 + '_' + pid2)
                else:
                    pairs.append(pid2 + '_' + pid1)
        Xq = np.array(Xq)
        Xp = np.array(Xp)
        Xq = torch.as_tensor(Xq, dtype=torch.float)
        Xp = torch.as_tensor(Xp, dtype=torch.float)
        return Xq, Xp, pairs



def query_embedder(query_list, embedding_model):
    query_embed = {}
    queries = []
    query_ids = []
    for q in query_list:
        hash = sha1(str.encode(q)).hexdigest()
        if 'Query:'+hash not in query_ids:
            query_ids.append('Query:'+hash)
            queries.append(q)
    model = SentenceTransformer(embedding_model)
    query_vecs = model.encode(queries, show_progress_bar=True)
    for i in range(len(query_ids)):
        query_embed[query_ids[i]] = {'query': queries[i], 'query_vec': query_vecs[i]}
    return query_embed

def rewrite_qry_attn_with_qryID(old_qry_attn_file, output_file):
    with open(old_qry_attn_file, 'r') as sq:
        lines = []
        for l in sq:
            elems = l.split('\t')
            lines.append(
                elems[0] + '\t' + 'Query:' + sha1(str.encode(elems[1])).hexdigest() + '\t' + elems[2] + '\t' + elems[3])
    with open(output_file, 'w') as out:
        for l in lines:
            out.write(l)

def read_art_qrels(art_qrels):
    page_paras = {}
    with open(art_qrels, 'r') as f:
        for l in f:
            q = l.split(' ')[0]
            p = l.split(' ')[2]
            if q not in page_paras.keys():
                page_paras[q] = [p]
            else:
                page_paras[q].append(p)
    return page_paras

def main():
    qry_attn_file = '/home/sk1105/sumanta/CATS_data/half-y1train-qry-attn.tsv'
    pids_npy = np.load('/home/sk1105/sumanta/CATS_data/half-y1train-qry-attn-paraids-sentwise.npy')
    pvecs_npy = np.load('/home/sk1105/sumanta/CATS_data/half-y1train-qry-attn-paravecs-sentwise.npy')
    qids_npy = np.load('/home/sk1105/sumanta/CATS_data/half-y1train-context-qids.npy')
    qvecs_npy = np.load('/home/sk1105/sumanta/CATS_data/half-y1train-context-qvecs.npy')
    qry_attn = []
    with open(qry_attn_file, 'r') as trf:
        f = True
        for l in trf:
            if f:
                f = False
                continue
            qry_attn.append(l.split('\t'))
    dat = InputSentenceCATSDatasetBuilder(qry_attn, pids_npy, pvecs_npy, qids_npy, qvecs_npy)
    Xq, Xp, y = dat.build_input_data()

if __name__ == '__main__':
    main()