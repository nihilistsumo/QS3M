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
from scipy.stats import ttest_rel

def eval_all_pairs(parapairs_data, model, test_pids_file, test_pvecs_file, test_qids_file,
                 test_qvecs_file, max_seq_len):
    test_pids = np.load(test_pids_file)
    test_pvecs = np.load(test_pvecs_file)
    test_qids = np.load(test_qids_file)
    test_qvecs = np.load(test_qvecs_file)
    model.cpu()
    with open(parapairs_data, 'r') as f:
        parapairs = json.load(f)
    qry_attn = []
    for page in parapairs.keys():
        qid = 'Query:' + sha1(str.encode(page)).hexdigest()
        for i in range(len(parapairs[page]['parapairs'])):
            p1 = parapairs[page]['parapairs'][i].split('_')[0]
            p2 = parapairs[page]['parapairs'][i].split('_')[1]
            qry_attn.append([qid, p1, p2, int(parapairs[page]['labels'][i])])
    test_data_builder = InputSentenceCATSDatasetBuilder(qry_attn, test_pids, test_pvecs, test_qids, test_qvecs, max_seq_len)
    pagewise_all_auc = {}
    for page in parapairs.keys():
        qry_attn_ts = []
        qid = 'Query:'+sha1(str.encode(page)).hexdigest()
        for i in range(len(parapairs[page]['parapairs'])):
            p1 = parapairs[page]['parapairs'][i].split('_')[0]
            p2 = parapairs[page]['parapairs'][i].split('_')[1]
            qry_attn_ts.append([qid, p1, p2, int(parapairs[page]['labels'][i])])

        X_test_q, X_test_p, y_test, pairs = test_data_builder.build_input_data(qry_attn_ts)
        if len(set(y_test.cpu().numpy())) < 2:
            continue
        ypred_test = model(X_test_q, X_test_p)
        test_auc = roc_auc_score(y_test.detach().cpu().numpy(), ypred_test.detach().cpu().numpy())
        print(page+' Method AUC: %.5f' % test_auc)
        pagewise_all_auc[page] = test_auc
    return pagewise_all_auc

'''
def eval_cluster(qry_attn_file_test, parapairs_data, model, test_pids_file, test_pvecs_file, test_qids_file,
                 test_qvecs_file, article_qrels, top_qrels, max_seq_len):

    test_pids = np.load(test_pids_file)
    test_pvecs = np.load(test_pvecs_file)
    test_qids = np.load(test_qids_file)
    test_qvecs = np.load(test_qvecs_file)

    qry_attn = []
    with open(qry_attn_file_test, 'r') as tsf:
        f = True
        for l in tsf:
            if f:
                f = False
                continue
            qry_attn.append(l.split('\t'))

    test_data_builder = InputSentenceCATSDatasetBuilder(qry_attn, test_pids, test_pvecs, test_qids, test_qvecs, max_seq_len)
    #X_test_q, X_test_p, y_test, _ = test_data_builder.build_input_data()

    model.cpu()
    
    #ypred_test = model(X_test_q, X_test_p)
    #mseloss = nn.MSELoss()
    #test_loss = mseloss(ypred_test, y_test)
    #test_auc = roc_auc_score(y_test.detach().cpu().numpy(), ypred_test.detach().cpu().numpy())
    #print('\n\nTest loss: %.5f, Test balanced auc: %.5f' % (test_loss.item(), test_auc))
    
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
        qry_attn_for_page = [d for d in qry_attn if d[0] == qid]
        X_test_q, X_test_p, y_test, pairs = test_data_builder.build_input_data(qry_attn_for_page)
        paras_in_pairs = set()
        for i in range(len(qry_attn_for_page)):
            p1 = qry_attn_for_page[i][1]
            p2 = qry_attn_for_page[i][2]
            paras_in_pairs.add(p1)
            paras_in_pairs.add(p2)
        #test_data_builder = InputSentenceCATSDatasetBuilder(qry_attn_ts, test_pids, test_pvecs, test_qids, test_qvecs, max_seq_len)
        #X_test_q, X_test_p, y_test, pairs = test_data_builder.build_input_data(qry_attn_ts)
        model.cpu()
        pair_scores = model(X_test_q, X_test_p)

        paralist = list(paras_in_pairs)
        paralist = page_paras[page]
        paralist.sort()
        true_labels = []
        for i in range(len(paralist)):
            true_labels.append(para_labels[paralist[i]])
        pair_scores = (pair_scores - torch.min(pair_scores)) / (torch.max(pair_scores) - torch.min(pair_scores))
        pair_score_dict = {}
        for i in range(len(pairs)):
            pair_score_dict[pairs[i]] = 1 - pair_scores[i].item()
        dist_mat = []
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

    #with open('/home/sk1105/sumanta/CATS_data/sentCats_y1train_hier.json', 'w') as f:
    #    json.dump(pagewise_ari_score, f)
    print('Mean ARI score: %.5f' % np.mean(np.array(list(pagewise_ari_score.values()))))
'''

def eval_cluster(qry_attn_file_test, parapair_file, model, test_pids_file, test_pvecs_file, test_pids_para_file, test_pvecs_para_file,
                 test_qids_file, test_qvecs_file, article_qrels, top_qrels, hier_qrels, max_seq_len):
    qry_attn_full = []
    with open(qry_attn_file_test, 'r') as tsf:
        f = True
        for l in tsf:
            if f:
                f = False
                continue
            qry_attn_full.append(l.split('\t'))
    test_pids = np.load(test_pids_file)
    test_pvecs = np.load(test_pvecs_file)
    test_pids_para = np.load(test_pids_para_file)
    test_pvecs_para = np.load(test_pvecs_para_file)
    test_qids = np.load(test_qids_file)
    test_qvecs = np.load(test_qvecs_file)

    test_data_builder = InputSentenceCATSDatasetBuilder(qry_attn_full, test_pids, test_pvecs, test_qids, test_qvecs, max_seq_len)
    test_data_builder_para = InputCATSDatasetBuilder(qry_attn_full, test_pids_para, test_pvecs_para, test_qids, test_qvecs)

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

    para_labels_hq = {}
    with open(hier_qrels, 'r') as f:
        for l in f:
            para = l.split(' ')[2]
            sec = l.split(' ')[0]
            para_labels_hq[para] = sec
    page_num_sections_hq = {}
    for page in page_paras.keys():
        paras = page_paras[page]
        sec = set()
        for p in paras:
            sec.add(para_labels_hq[p])
        page_num_sections_hq[page] = len(sec)

    pagewise_ari_score = {}
    pagewise_hq_ari_score = {}
    pagewise_euc_ari_score = {}
    pagewise_hq_euc_ari_score = {}
    anchor_auc = []
    cand_auc = []
    cos_auc = []
    anchor_ari_scores = []
    cand_ari_scores = []
    anchor_ari_scores_hq = []
    cand_ari_scores_hq = []

    for page in page_paras.keys():
        qid = 'Query:'+sha1(str.encode(page)).hexdigest()
        if qid not in test_data_builder_para.query_vecs.keys():
            print(qid + ' not present in query vecs dict')
        else:
            qry_attn_for_page = [d for d in qry_attn_full if d[0]==qid]
            #test_data_builder_for_page = InputCATSDatasetBuilder(qry_attn_for_page, test_pids, test_pvecs, test_qids, test_qvecs)
            X_q_page, X_p_page, y_page, pairs_page = test_data_builder.build_input_data(qry_attn_for_page)
            ypred_test_page = model(X_q_page, X_p_page)
            test_auc_page = roc_auc_score(y_page.detach().cpu().numpy(), ypred_test_page.detach().cpu().numpy())

            X_para_page, y_para_page = test_data_builder_para.build_input_data(qry_attn_for_page)
            y_euclid_page = torch.sqrt(torch.sum((X_para_page[:, 768:768 * 2] - X_para_page[:, 768 * 2:]) ** 2, 1)).numpy()
            y_euclid_page = 1 - (y_euclid_page - np.min(y_euclid_page)) / (np.max(y_euclid_page) - np.min(y_euclid_page))
            euclid_auc_page = roc_auc_score(y_para_page, y_euclid_page)
            anchor_auc.append(euclid_auc_page)
            cand_auc.append(test_auc_page)

            paralist = page_paras[page]
            true_labels = []
            true_labels_hq = []
            for i in range(len(paralist)):
                true_labels.append(para_labels[paralist[i]])
                true_labels_hq.append(para_labels_hq[paralist[i]])
            X_page, parapairs = test_data_builder_para.build_cluster_data(qid, paralist)
            pair_euclid_scores = torch.sqrt(torch.sum((X_page[:, 768:768 * 2] - X_page[:, 768 * 2:])**2, 1)).numpy()
            pair_euclid_scores = (pair_euclid_scores - np.min(pair_euclid_scores)) / (np.max(pair_euclid_scores) - np.min(pair_euclid_scores))
            pair_scores = (ypred_test_page - torch.min(ypred_test_page)) / (torch.max(ypred_test_page) - torch.min(ypred_test_page))
            pair_score_dict = {}
            pair_euclid_score_dict = {}
            for pp in parapairs:
                if pp in pairs_page:
                    pair_score_dict[pp] = 1-pair_scores[pairs_page.index(pp)].item()
                else:
                    print(pp+' not in list')
                    pair_score_dict[pp] = 1.0
                pair_euclid_score_dict[pp] = pair_euclid_scores[parapairs.index(pp)]
            dist_mat = []
            dist_euc_mat = []
            paralist.sort()
            for i in range(len(paralist)):
                r = []
                reuc = []
                for j in range(len(paralist)):
                    if i == j:
                        r.append(0.0)
                        reuc.append(0.0)
                    elif i < j:
                        r.append(pair_score_dict[paralist[i]+ '_' + paralist[j]])
                        reuc.append(pair_euclid_score_dict[paralist[i] + '_' + paralist[j]])
                    else:
                        r.append(pair_score_dict[paralist[j] + '_' + paralist[i]])
                        reuc.append(pair_euclid_score_dict[paralist[j] + '_' + paralist[i]])
                dist_mat.append(r)
                dist_euc_mat.append(reuc)

            cl = AgglomerativeClustering(n_clusters=page_num_sections[page], affinity='precomputed', linkage='average')
            cl_labels = cl.fit_predict(dist_mat)
            cl_euclid_labels = cl.fit_predict(dist_euc_mat)

            cl_hq = AgglomerativeClustering(n_clusters=page_num_sections_hq[page], affinity='precomputed', linkage='average')
            cl_labels_hq = cl_hq.fit_predict(dist_mat)
            cl_euclid_labels_hq = cl_hq.fit_predict(dist_euc_mat)

            ari_score = adjusted_rand_score(true_labels, cl_labels)
            ari_score_hq = adjusted_rand_score(true_labels_hq, cl_labels_hq)
            ari_euc_score = adjusted_rand_score(true_labels, cl_euclid_labels)
            ari_euc_score_hq = adjusted_rand_score(true_labels_hq, cl_euclid_labels_hq)
            print(page+' ARI: %.5f, Euclid ARI: %.5f' %
                  (ari_score, ari_euc_score))
            pagewise_ari_score[page] = ari_score
            pagewise_euc_ari_score[page] = ari_euc_score
            pagewise_hq_ari_score[page] = ari_score_hq
            pagewise_euc_ari_score[page] = ari_euc_score
            pagewise_hq_euc_ari_score[page] = ari_euc_score_hq
            anchor_ari_scores.append(ari_euc_score)
            cand_ari_scores.append(ari_score)
            anchor_ari_scores_hq.append(ari_euc_score_hq)
            cand_ari_scores_hq.append(ari_score_hq)

    test_auc = np.mean(np.array(cand_auc))
    euclid_auc = np.mean(np.array(anchor_auc))
    paired_ttest_auc = ttest_rel(anchor_auc, cand_auc)
    mean_ari = np.mean(np.array(list(pagewise_ari_score.values())))
    mean_euc_ari = np.mean(np.array(list(pagewise_euc_ari_score.values())))
    mean_ari_hq = np.mean(np.array(list(pagewise_hq_ari_score.values())))
    mean_euc_ari_hq = np.mean(np.array(list(pagewise_hq_euc_ari_score.values())))

    print('Mean ARI score: %.5f' % mean_ari)
    print('Mean Euclid ARI score: %.5f' % mean_euc_ari)
    paired_ttest_ari = ttest_rel(anchor_ari_scores, cand_ari_scores)
    print('Paired ttest: %.5f, p val: %.5f' % (paired_ttest_ari[0], paired_ttest_ari[1]))
    print('Mean hq ARI score: %.5f' % mean_ari_hq)
    print('Mean hq Euclid ARI score: %.5f' % mean_euc_ari_hq)
    paired_ttest_ari_hq = ttest_rel(anchor_ari_scores_hq, cand_ari_scores_hq)
    print('Paired ttest hq: %.5f, p val: %.5f' % (paired_ttest_ari_hq[0], paired_ttest_ari_hq[1]))
    #with open('/home/sk1105/sumanta/CATS_data/anchor_euc_y1test_hier.json', 'w') as f:
    #    json.dump(pagewise_euc_ari_score, f)
    return test_auc, euclid_auc, cos_auc, mean_ari, mean_euc_ari, mean_ari_hq, mean_euc_ari_hq, \
           paired_ttest_ari, paired_ttest_ari_hq, paired_ttest_auc

def main():

    parser = argparse.ArgumentParser(description='Run CATS model')

    parser.add_argument('-dd', '--data_dir', default="/home/sk1105/sumanta/CATS_data/")
    parser.add_argument('-qt', '--qry_attn_test', default="by1test-qry-attn-bal-allpos-for-eval.tsv")
    parser.add_argument('-aql', '--art_qrels', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-article.qrels")
    parser.add_argument('-tql', '--top_qrels',
                        default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-toplevel.qrels")
    parser.add_argument('-hql', '--hier_qrels', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-hierarchical.qrels")
    parser.add_argument('-pp', '--parapairs', default="/home/sk1105/sumanta/Mule-data/input_data_v2/pairs/test-cleaned-parapairs/by1-test-cleaned.parapairs.json")
    parser.add_argument('-tp', '--test_pids', default="by1test-all-pids-sentwise.npy")
    parser.add_argument('-tv', '--test_pvecs', default="by1test-all-paravecs-sentwise.npy")
    parser.add_argument('-tpp', '--test_pids_para', default="by1test-all-pids.npy")
    parser.add_argument('-tvp', '--test_pvecs_para', default="by1test-all-paravecs.npy")
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
    parser.add_argument('-hq', '--hier_qrels', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-toplevel.qrels")
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
    eval_cluster(dat+args.qry_attn_test, args.parapairs, model, dat+args.test_pids, dat+args.test_pvecs,
                 dat+args.test_pids_para, dat+args.test_pvecs_para, dat+args.test_qids, dat+args.test_qvecs,
                 args.art_qrels, args.top_qrels, args.hier_qrels, args.max_seq)

if __name__ == '__main__':
    main()