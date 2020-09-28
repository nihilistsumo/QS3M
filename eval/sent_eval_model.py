from model.layers import CATS, CATS_Scaled, CATS_QueryScaler, CATS_manhattan
from model.models import CATSSimilarityModel
from model.sent_models import CATSSentenceModel
from data.utils import InputCATSDatasetBuilder, read_art_qrels, InputSentenceCATSDatasetBuilder
import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.random import seed
seed(42)
from hashlib import sha1
from sklearn.metrics import roc_auc_score, adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import argparse
import math
import time
import json

def eval_all_pairs(parapairs_data, model, test_pids_file, test_pvecs_file, test_qids_file,
                 test_qvecs_file, max_seq_len):
    qry_attn_ts = []
    with open(parapairs_data, 'r') as f:
        parapairs = json.load(f)
    for page in parapairs.keys():
        qid = 'Query:'+sha1(str.encode(page)).hexdigest()
        for i in range(len(parapairs[page]['parapairs'])):
            p1 = parapairs[page]['parapairs'][i].split('_')[0]
            p2 = parapairs[page]['parapairs'][i].split('_')[1]
            qry_attn_ts.append([qid, p1, p2, int(parapairs[page]['labels'][i])])
    test_pids = np.load(test_pids_file)
    test_pvecs = np.load(test_pvecs_file)
    test_qids = np.load(test_qids_file)
    test_qvecs = np.load(test_qvecs_file)
    test_data_builder = InputSentenceCATSDatasetBuilder(qry_attn_ts, test_pids, test_pvecs, test_qids, test_qvecs, max_seq_len)
    X_test_q, X_test_p, y_test, pairs = test_data_builder.build_input_data()

    model.cpu()
    ypred_test = model(X_test_q, X_test_p)
    mseloss = nn.MSELoss()
    test_loss = mseloss(ypred_test, y_test)
    test_auc = roc_auc_score(y_test.detach().cpu().numpy(), ypred_test.detach().cpu().numpy())
    print('\n\nTest loss: %.5f, Test all pairs auc: %.5f' % (test_loss.item(), test_auc))

def eval_cluster(parapairs_data, model, test_pids_file, test_pvecs_file, test_qids_file,
                 test_qvecs_file, article_qrels, top_qrels, max_seq_len):

    test_pids = np.load(test_pids_file)
    test_pvecs = np.load(test_pvecs_file)
    test_qids = np.load(test_qids_file)
    test_qvecs = np.load(test_qvecs_file)

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
    with open(parapairs_data, 'r') as f:
        parapairs = json.load(f)
    for page in parapairs.keys():
        qry_attn_ts = []
        qid = 'Query:'+sha1(str.encode(page)).hexdigest()
        paras_in_pairs = set()
        for i in range(len(parapairs[page]['parapairs'])):
            p1 = parapairs[page]['parapairs'][i].split('_')[0]
            p2 = parapairs[page]['parapairs'][i].split('_')[1]
            paras_in_pairs.add(p1)
            paras_in_pairs.add(p2)
            qry_attn_ts.append([qid, p1, p2, int(parapairs[page]['labels'][i])])
        test_data_builder = InputSentenceCATSDatasetBuilder(qry_attn_ts, test_pids, test_pvecs, test_qids, test_qvecs,
                                                            max_seq_len)
        X_test_q, X_test_p, y_test, pairs = test_data_builder.build_input_data()
        model.cpu()
        pair_scores = model(X_test_q, X_test_p)

        paralist = list(paras_in_pairs)

        true_labels = []
        for i in range(len(paralist)):
            true_labels.append(para_labels[paralist[i]])
        pair_scores = (pair_scores - torch.min(pair_scores)) / (torch.max(pair_scores) - torch.min(pair_scores))
        pair_score_dict = {}
        for i in range(len(pairs)):
            pair_score_dict[pairs[i]] = 1 - pair_scores[i].item()
        dist_mat = []
        paralist.sort()
        for i in range(len(paralist)):
            r = []
            for j in range(len(paralist)):
                if i == j:
                    r.append(0.0)
                elif paralist[i] < paralist[j]:
                    r.append(pair_score_dict[paralist[i] + '_' + paralist[j]])
                else:
                    r.append(pair_score_dict[paralist[j] + '_' + paralist[i]])
            dist_mat.append(r)

        cl = AgglomerativeClustering(n_clusters=page_num_sections[page], affinity='precomputed', linkage='average')
        # cl = AgglomerativeClustering(n_clusters=8, affinity='precomputed', linkage='average')
        # cl = DBSCAN(eps=0.7, min_samples=3)
        cl_labels = cl.fit_predict(dist_mat)
        ari_score = adjusted_rand_score(true_labels, cl_labels)
        print(page + ' ARI: %.5f' % ari_score)
        pagewise_ari_score[page] = ari_score

    print('Mean ARI score: %.5f' % np.mean(np.array(list(pagewise_ari_score.values()))))

def main():

    parser = argparse.ArgumentParser(description='Run CATS model')

    parser.add_argument('-dd', '--data_dir', default="/home/sk1105/sumanta/CATS_data/")
    parser.add_argument('-qt', '--qry_attn_test', default="by1test-qry-attn-bal-allpos-for-eval.tsv")
    parser.add_argument('-aq', '--art_qrels', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-article.qrels")
    parser.add_argument('-hq', '--hier_qrels', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-hierarchical.qrels")
    parser.add_argument('-pp', '--parapairs', default="/home/sk1105/sumanta/Mule-data/input_data_v2/pairs/test-cleaned-parapairs/by1-test-cleaned.parapairs.json")
    parser.add_argument('-tp', '--test_pids', default="by1test-all-pids-sentwise.npy")
    parser.add_argument('-tv', '--test_pvecs', default="by1test-all-paravecs-sentwise.npy")
    parser.add_argument('-tq', '--test_qids', default="by1test-context-qids.npy")
    parser.add_argument('-tqv', '--test_qvecs', default="by1test-context-qvecs.npy")
    parser.add_argument('-cp', '--cats_path', default="/home/sk1105/sumanta/CATS/saved_models/cats_leadpara_b32_l0.00001_i3.model")
    parser.add_argument('-seq', '--max_seq', type=int, default=10)
    parser.add_argument('-pn', '--param_n', type=int, default=32)
    parser.add_argument('-mt', '--model_type', default="fcats")
    parser.add_argument('-mp', '--model_path', default="/home/sk1105/sumanta/CATS/saved_models/sentcats_maxlen_10_leadpara_b32_l0.0001_i6.model")

    '''
    parser.add_argument('-dd', '--data_dir', default="/home/sk1105/sumanta/CATS_data/")
    parser.add_argument('-qt', '--qry_attn_test', default="by1train-qry-attn-bal-allpos.tsv")
    parser.add_argument('-aq', '--art_qrels', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-article.qrels")
    parser.add_argument('-hq', '--hier_qrels', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-hierarchical.qrels")
    parser.add_argument('-pp', '--parapairs',
                        default="/home/sk1105/sumanta/Mule-data/input_data_v2/pairs/train-cleaned-parapairs/by1-train-cleaned.parapairs.json")
    parser.add_argument('-tp', '--test_pids', default="by1train-all-pids-sentwise.npy")
    parser.add_argument('-tv', '--test_pvecs', default="by1train-all-paravecs-sentwise.npy")
    parser.add_argument('-tq', '--test_qids', default="by1train-context-qids.npy")
    parser.add_argument('-tqv', '--test_qvecs', default="by1train-context-qvecs.npy")
    parser.add_argument('-cp', '--cats_path', default="/home/sk1105/sumanta/CATS/saved_models/cats_leadpara_b32_l0.00001_i3.model")
    parser.add_argument('-seq', '--max_seq', type=int, default=10)
    parser.add_argument('-pn', '--param_n', type=int, default=32)
    parser.add_argument('-mt', '--model_type', default="fcats")
    parser.add_argument('-mp', '--model_path', default="/home/sk1105/sumanta/CATS/saved_models/sentcats_maxlen_10_leadpara_b32_l0.0001_i6.model")

    
    parser.add_argument('-dd', '--data_dir', default="/home/sk1105/sumanta/CATS_data/")
    parser.add_argument('-qt', '--qry_attn_test', default="by2test-qry-attn-bal-allpos.tsv")
    parser.add_argument('-aq', '--art_qrels',
                        default="/home/sk1105/sumanta/trec_dataset/benchmarkY2/benchmarkY2test-goldpassages.onlywiki.article.nodup.qrels")
    parser.add_argument('-hq', '--hier_qrels',
                        default="/home/sk1105/sumanta/trec_dataset/benchmarkY2/benchmarkY2test-goldpassages.onlywiki.toplevel.nodup.qrels")
    parser.add_argument('-tp', '--test_pids', default="by2test-all-pids.npy")
    parser.add_argument('-tv', '--test_pvecs', default="by2test-all-paravecs.npy")
    parser.add_argument('-tq', '--test_qids', default="by2test-context-qids.npy")
    parser.add_argument('-tqv', '--test_qvecs', default="by2test-context-qvecs.npy")
    parser.add_argument('-cp', '--cats_path', default="/home/sk1105/sumanta/CATS/saved_models/cats_leadpara_b32_l0.00001_i3.model")
    parser.add_argument('-mt', '--model_type', default="cats")
    parser.add_argument('-mp', '--model_path',
                        default="/home/sk1105/sumanta/CATS/saved_models/cats_leadpara_b32_l0.00001_i3.model")

    '''
    args = parser.parse_args()
    dat = args.data_dir

    model = CATSSentenceModel(768, args.param_n, args.model_type, args.cats_path)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    eval_all_pairs(args.parapairs, model, dat+args.test_pids, dat+args.test_pvecs, dat+args.test_qids,
                 dat+args.test_qvecs, args.max_seq)
    eval_cluster(args.parapairs, model, dat+args.test_pids, dat+args.test_pvecs, dat+args.test_qids,
                 dat+args.test_qvecs, args.art_qrels, args.hier_qrels, args.max_seq)

if __name__ == '__main__':
    main()