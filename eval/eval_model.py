from model.layers import CATS, CATS_Scaled, CATS_QueryScaler, CATS_manhattan
from model.models import CATSSimilarityModel
from data.utils import InputCATSDatasetBuilder, read_art_qrels
import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.random import seed
seed(42)
from hashlib import sha1
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
import argparse
import math
import time

def eval_cluster(model_path, model_type, qry_attn_file_test, test_pids_file, test_pvecs_file, test_qids_file,
                 test_qvecs_file, article_qrels, top_qrels):
    model = CATSSimilarityModel(768)
    if model_type == 'triam':
        model.cats = CATS(768)
    elif model_type == 'qscale':
        model.cats = CATS_QueryScaler(768)
    else:
        print('Wrong model type')
    model.load_state_dict(torch.load(model_path))
    model.eval()
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

    model.cpu()
    ypred_test = model(X_test)
    mseloss = nn.MSELoss()
    test_loss = mseloss(ypred_test, y_test)
    test_auc = roc_auc_score(y_test.detach().cpu().numpy(), ypred_test.detach().cpu().numpy())
    print('\n\nTest loss: %.5f, Test auc: %.5f' % (test_loss.item(), test_auc))

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    y_cos = cos(X_test[:, 768:768 * 2], X_test[:, 768 * 2:])
    cos_auc = roc_auc_score(y_test, y_cos)
    print('Test cosine auc: ' + str(cos_auc))
    y_euclid = torch.sqrt(torch.sum((X_test[:, 768:768 * 2] - X_test[:, 768 * 2:])**2, 1)).numpy()
    y_euclid = 1 - (y_euclid - np.min(y_euclid)) / (np.max(y_euclid) - np.min(y_euclid))
    euclid_auc = roc_auc_score(y_test, y_euclid)
    print('Test euclidean auc: ' + str(euclid_auc))

    page_paras = read_art_qrels(article_qrels)
    para_labels = {}
    with open(top_qrels, 'r') as f:
        for l in f:
            para = l.split(' ')[2]
            sec = l.split(' ')[0]
            para_labels[para] = sec
    page_num_sections = {}
    for page in page_paras.keys():
        paras = page_paras[page]
        sec = set()
        for p in paras:
            sec.add(para_labels[p])
        page_num_sections[page] = len(sec)

    pagewise_ari_score = {}
    pagewise_base_ari_score = {}
    for page in page_paras.keys():
        print('Going to cluster '+page)
        qid = 'Query:'+sha1(str.encode(page)).hexdigest()
        if qid not in test_data_builder.query_vecs.keys():
            print(qid + ' not present in query vecs dict')
        else:
            paralist = page_paras[page]

            true_labels = []
            for i in range(len(paralist)):
                true_labels.append(para_labels[paralist[i]])
            X_page, parapairs = test_data_builder.build_cluster_data(qid, paralist)
            pair_baseline_scores = cos(X_page[:, 768:768 * 2], X_page[:, 768 * 2:])
            pair_scores = model(X_page)
            pair_scores = (pair_scores - torch.min(pair_scores))/(torch.max(pair_scores) - torch.min(pair_scores))
            pair_baseline_scores = (pair_baseline_scores - torch.min(pair_baseline_scores)) / (torch.max(pair_baseline_scores) - torch.min(pair_baseline_scores))
            pair_score_dict = {}
            pair_baseline_score_dict = {}
            for i in range(len(parapairs)):
                pair_score_dict[parapairs[i]] = 1-pair_scores[i].item()
                pair_baseline_score_dict[parapairs[i]] = 1-pair_baseline_scores[i]
            dist_mat = []
            dist_base_mat = []
            for i in range(len(paralist)):
                r = []
                rbase = []
                for j in range(len(paralist)):
                    if i == j:
                        r.append(0.0)
                        rbase.append(0.0)
                    elif paralist[i]+'_'+paralist[j] in parapairs:
                        r.append(pair_score_dict[paralist[i]+ '_' + paralist[j]])
                        rbase.append(pair_baseline_score_dict[paralist[i]+ '_' + paralist[j]])
                    else:
                        r.append(pair_score_dict[paralist[j] + '_' + paralist[i]])
                        rbase.append(pair_baseline_score_dict[paralist[j] + '_' + paralist[i]])
                dist_mat.append(r)
                dist_base_mat.append(rbase)
            cl = AgglomerativeClustering(n_clusters=page_num_sections[page], affinity='precomputed', linkage='complete')
            cl_labels = cl.fit_predict(dist_mat)
            cl_base_labels = cl.fit_predict(dist_base_mat)
            ari_score = adjusted_rand_score(true_labels, cl_labels)
            ari_base_score = adjusted_rand_score(true_labels, cl_base_labels)
            print(page+' ARI score: '+str(ari_score), ', Baseline score: '+str(ari_base_score))
            pagewise_ari_score[page] = ari_score
            pagewise_base_ari_score[page] = ari_base_score
    print('Mean ARI score: '+str(np.mean(np.array(list(pagewise_ari_score.values())))))
    print('Mean Baseline ARI score: ' + str(np.mean(np.array(list(pagewise_base_ari_score.values())))))

def main():
    parser = argparse.ArgumentParser(description='Run CATS model')
    parser.add_argument('-dd', '--data_dir', default="/home/sk1105/sumanta/CATS_data/")
    parser.add_argument('-qt', '--qry_attn_test', default="by1test-qry-attn-bal-allpos-for-eval.tsv")
    parser.add_argument('-aq', '--art_qrels', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-article.qrels")
    parser.add_argument('-hq', '--hier_qrels', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-hierarchical.qrels")
    parser.add_argument('-tp', '--test_pids', default="by1test-all-pids.npy")
    parser.add_argument('-tv', '--test_pvecs', default="by1test-all-paravecs.npy")
    parser.add_argument('-tq', '--test_qids', default="by1test-context-qids.npy")
    parser.add_argument('-tqv', '--test_qvecs', default="by1test-context-qvecs.npy")
    parser.add_argument('-mt', '--model_type', default="triam")
    parser.add_argument('-mp', '--model_path', default="/home/sk1105/sumanta/CATS/saved_models/cats_2triamese_layer_b32_l0.00001_i7.model")

    args = parser.parse_args()
    dat = args.data_dir

    eval_cluster(args.model_path, args.model_type, dat+args.qry_attn_test, dat+args.test_pids, dat+args.test_pvecs, dat+args.test_qids,
                 dat+args.test_qvecs, args.art_qrels, args.hier_qrels)

if __name__ == '__main__':
    main()