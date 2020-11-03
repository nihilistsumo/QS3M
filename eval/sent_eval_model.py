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
from sklearn.metrics import roc_auc_score, adjusted_rand_score, f1_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import argparse
import math
import time
import json
from scipy.stats import ttest_rel

def calc_f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    yp = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
    yp = np.array([1.0 if d > 0.5 else 0.0 for d in yp])
    test_f1 = f1_score(y_true, yp)
    return test_f1

def eval_all_pairs(parapairs_data, model, test_pids_file, test_pvecs_file, test_pids_para_file, test_pvecs_para_file,
                   test_qids_file, test_qvecs_file, max_seq_len):
    test_pids = np.load(test_pids_file)
    test_pvecs = np.load(test_pvecs_file)
    test_pids_para = np.load(test_pids_para_file)
    test_pvecs_para = np.load(test_pvecs_para_file)
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
    test_data_builder_para = InputCATSDatasetBuilder(qry_attn, test_pids_para, test_pvecs_para, test_qids, test_qvecs)
    cand_auc = []
    cand_f1 = []
    anchor_auc = []
    anchor_f1 = []
    for page in parapairs.keys():
        qry_attn_ts = []
        qid = 'Query:'+sha1(str.encode(page)).hexdigest()
        for i in range(len(parapairs[page]['parapairs'])):
            p1 = parapairs[page]['parapairs'][i].split('_')[0]
            p2 = parapairs[page]['parapairs'][i].split('_')[1]
            qry_attn_ts.append([qid, p1, p2, int(parapairs[page]['labels'][i])])

        X_test_q, X_test_p, y_test, pairs = test_data_builder.build_input_data(qry_attn_ts)
        X_test_para, y_test_para = test_data_builder_para.build_input_data(qry_attn_ts)
        if len(set(y_test.cpu().numpy())) < 2:
            continue
        ypred_test = model(X_test_q, X_test_p)
        y_euclid = torch.sqrt(torch.sum((X_test_para[:, 768:768 * 2] - X_test_para[:, 768 * 2:]) ** 2, 1)).numpy()
        y_euclid = 1 - (y_euclid - np.min(y_euclid)) / (np.max(y_euclid) - np.min(y_euclid))

        test_auc = roc_auc_score(y_test.detach().cpu().numpy(), ypred_test.detach().cpu().numpy())
        test_fm = calc_f1(y_test.detach().cpu().numpy(), ypred_test.detach().cpu().numpy())
        euclid_auc = roc_auc_score(y_test_para, y_euclid)
        euclid_fm = calc_f1(y_test_para, y_euclid)
        print(page+' Method all-pair AUC: %.5f, F1: %.5f' % (test_auc, test_fm))
        cand_auc.append(test_auc)
        cand_f1.append(test_fm)
        anchor_auc.append(euclid_auc)
        anchor_f1.append(euclid_fm)
    paired_ttest = ttest_rel(anchor_auc, cand_auc)
    paired_ttest_f1 = ttest_rel(anchor_f1, cand_f1)
    all_auc = np.mean(np.array(cand_auc))
    all_f1 = np.mean(np.array(cand_f1))
    euc_auc = np.mean(np.array(anchor_auc))
    euc_f1 = np.mean(np.array(anchor_f1))
    return all_auc, euc_auc, paired_ttest, all_f1, euc_f1, paired_ttest_f1

def eval_cluster(qry_attn_file_test, model, test_pids_file, test_pvecs_file, test_pids_para_file, test_pvecs_para_file,
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
    anchor_f1 = []
    cand_f1 = []
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
            X_q_page, X_p_page, y_page, _ = test_data_builder.build_input_data(qry_attn_for_page)
            ypred_test_page = model(X_q_page, X_p_page)
            test_auc_page = roc_auc_score(y_page.detach().cpu().numpy(), ypred_test_page.detach().cpu().numpy())
            test_f1_page = calc_f1(y_page.detach().cpu().numpy(), ypred_test_page.detach().cpu().numpy())

            X_para_page, y_para_page = test_data_builder_para.build_input_data(qry_attn_for_page)
            y_euclid_page = torch.sqrt(torch.sum((X_para_page[:, 768:768 * 2] - X_para_page[:, 768 * 2:]) ** 2, 1)).numpy()
            y_euclid_page = 1 - (y_euclid_page - np.min(y_euclid_page)) / (np.max(y_euclid_page) - np.min(y_euclid_page))
            euclid_auc_page = roc_auc_score(y_para_page, y_euclid_page)
            euclid_f1_page = calc_f1(y_para_page, y_euclid_page)

            anchor_auc.append(euclid_auc_page)
            cand_auc.append(test_auc_page)
            anchor_f1.append(euclid_f1_page)
            cand_f1.append(test_f1_page)

            paralist = page_paras[page]
            true_labels = []
            true_labels_hq = []
            paralist.sort()
            for i in range(len(paralist)):
                true_labels.append(para_labels[paralist[i]])
                true_labels_hq.append(para_labels_hq[paralist[i]])

            X_q, X_p, pairs = test_data_builder.build_cluster_data(qid, paralist)
            pair_scores = model(X_q, X_p)
            pair_scores = (pair_scores - torch.min(pair_scores)) / (torch.max(pair_scores) - torch.min(pair_scores))

            X_page, parapairs = test_data_builder_para.build_cluster_data(qid, paralist)
            pair_euclid_scores = torch.sqrt(torch.sum((X_page[:, 768:768 * 2] - X_page[:, 768 * 2:])**2, 1)).numpy()
            pair_euclid_scores = (pair_euclid_scores - np.min(pair_euclid_scores)) / (np.max(pair_euclid_scores) - np.min(pair_euclid_scores))

            pair_score_dict = {}
            pair_euclid_score_dict = {}
            for pp in parapairs:
                pair_score_dict[pp] = 1-pair_scores[pairs.index(pp)].item()
                pair_euclid_score_dict[pp] = pair_euclid_scores[parapairs.index(pp)]
            dist_mat = []
            dist_euc_mat = []

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
            print(page+' Method bal AUC: %.5f, F1: %.5f, ARI: %.5f, Euclid bal AUC: %.5f, F1: %.5f, ARI: %.5f' %
                  (test_auc_page, test_f1_page, ari_score, euclid_auc_page, euclid_f1_page, ari_euc_score))
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
    test_f1 = np.mean(np.array(cand_f1))
    euclid_f1 = np.mean(np.array(anchor_auc))
    paired_ttest_f1 = ttest_rel(anchor_f1, cand_f1)
    mean_ari = np.mean(np.array(list(pagewise_ari_score.values())))
    mean_euc_ari = np.mean(np.array(list(pagewise_euc_ari_score.values())))
    mean_ari_hq = np.mean(np.array(list(pagewise_hq_ari_score.values())))
    mean_euc_ari_hq = np.mean(np.array(list(pagewise_hq_euc_ari_score.values())))
    '''
    print('Mean ARI score: %.5f' % mean_ari)
    print('Mean Euclid ARI score: %.5f' % mean_euc_ari)
    
    print('Paired ttest: %.5f, p val: %.5f' % (paired_ttest_ari[0], paired_ttest_ari[1]))
    print('Mean hq ARI score: %.5f' % mean_ari_hq)
    print('Mean hq Euclid ARI score: %.5f' % mean_euc_ari_hq)
    
    print('Paired ttest hq: %.5f, p val: %.5f' % (paired_ttest_ari_hq[0], paired_ttest_ari_hq[1]))
    '''
    paired_ttest_ari = ttest_rel(anchor_ari_scores, cand_ari_scores)
    paired_ttest_ari_hq = ttest_rel(anchor_ari_scores_hq, cand_ari_scores_hq)
    return test_auc, euclid_auc, mean_ari, mean_euc_ari, mean_ari_hq, mean_euc_ari_hq, \
           paired_ttest_ari, paired_ttest_ari_hq, paired_ttest_auc, test_f1, euclid_f1, paired_ttest_f1

def main():

    parser = argparse.ArgumentParser(description='Run CATS model')

    parser.add_argument('-dd', '--data_dir', default="/home/sk1105/sumanta/new_cats_data/")

    parser.add_argument('-qt1', '--qry_attn_test1', default="by1test-qry-attn-bal-allpos.tsv")
    parser.add_argument('-aql1', '--art_qrels1', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-article.qrels")
    parser.add_argument('-tql1', '--top_qrels1', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-toplevel.qrels")
    parser.add_argument('-hql1', '--hier_qrels1', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-hierarchical.qrels")
    parser.add_argument('-pp1', '--parapairs1', default="/home/sk1105/sumanta/Mule-data/input_data_v2/pairs/test-cleaned-parapairs/by1-test-cleaned.parapairs.json")
    parser.add_argument('-tp1', '--test_pids1', default="by1test-all-pids-sentwise.npy")
    parser.add_argument('-tv1', '--test_pvecs1', default="by1test-all-paravecs-sentwise.npy")
    parser.add_argument('-tpp1', '--test_pids_para1', default="by1test-all-pids.npy")
    parser.add_argument('-tvp1', '--test_pvecs_para1', default="by1test-all-paravecs.npy")
    parser.add_argument('-tq1', '--test_qids1', default="by1test-context-title-qids.npy") #change
    parser.add_argument('-tqv1', '--test_qvecs1', default="by1test-context-title-qvecs.npy") #change

    parser.add_argument('-qt2', '--qry_attn_test2', default="by1train-qry-attn-bal-allpos.tsv")
    parser.add_argument('-aql2', '--art_qrels2', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-article.qrels")
    parser.add_argument('-tql2', '--top_qrels2', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-toplevel.qrels")
    parser.add_argument('-hql2', '--hier_qrels2', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-hierarchical.qrels")
    parser.add_argument('-pp2', '--parapairs2', default="/home/sk1105/sumanta/Mule-data/input_data_v2/pairs/train-cleaned-parapairs/by1-train-cleaned.parapairs.json")
    parser.add_argument('-tp2', '--test_pids2', default="by1train-all-pids-sentwise.npy")
    parser.add_argument('-tv2', '--test_pvecs2', default="by1train-all-paravecs-sentwise.npy")
    parser.add_argument('-tpp2', '--test_pids_para2', default="by1train-all-pids.npy")
    parser.add_argument('-tvp2', '--test_pvecs_para2', default="by1train-all-paravecs.npy")
    parser.add_argument('-tq2', '--test_qids2', default="by1train-context-title-qids.npy") #change
    parser.add_argument('-tqv2', '--test_qvecs2', default="by1train-context-title-qvecs.npy") #change

    parser.add_argument('-cp', '--cats_path', default="/home/sk1105/sumanta/cats_deploy/model/saved_models/cats_title_b32_l0.00001_i3.model") #change
    parser.add_argument('-seq', '--max_seq', type=int, default=10)
    parser.add_argument('-pn', '--param_n', type=int, default=32)
    parser.add_argument('-mt', '--model_type', default="fcats")
    parser.add_argument('-mp', '--model_path', default="/home/sk1105/sumanta/cats_deploy/model/saved_models/sentcats_maxlen_10_title_b32_l0.0001_i6.model") #change

    '''
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
    print("\nPagewise benchmark Y1 test")
    print("==========================")
    all_auc1, euc_auc1, ttest_auc1, all_fm1, all_euc_fm1, ttest_fm1 = eval_all_pairs(args.parapairs1, model, dat+args.test_pids1,
                                                       dat+args.test_pvecs1, dat+args.test_pids_para1,
                                                       dat+args.test_pvecs_para1, dat+args.test_qids1,
                                                       dat+args.test_qvecs1, args.max_seq)
    bal_auc1, bal_euc_auc1, mean_ari1, mean_euc_ari1, mean_ari_hq1, mean_euc_ari_hq1, paired_ttest_ari1, \
    paired_ttest_ari_hq1, ttest_bal_auc1, bal_fm1, bal_euc_fm1, ttest_bal_fm1 = eval_cluster(dat+args.qry_attn_test1, model, dat+args.test_pids1,
                                                           dat+args.test_pvecs1, dat+args.test_pids_para1,
                                                           dat+args.test_pvecs_para1, dat+args.test_qids1,
                                                           dat+args.test_qvecs1, args.art_qrels1, args.top_qrels1,
                                                           args.hier_qrels1, args.max_seq)

    print("\nPagewise benchmark Y1 train")
    print("===========================")
    all_auc2, euc_auc2, ttest_auc2, all_fm2, all_euc_fm2, ttest_fm2 = eval_all_pairs(args.parapairs2, model, dat + args.test_pids2,
                                                       dat + args.test_pvecs2, dat + args.test_pids_para2,
                                                       dat + args.test_pvecs_para2, dat + args.test_qids2,
                                                       dat + args.test_qvecs2, args.max_seq)
    bal_auc2, bal_euc_auc2, mean_ari2, mean_euc_ari2, mean_ari_hq2, mean_euc_ari_hq2, paired_ttest_ari2, \
    paired_ttest_ari_hq2, ttest_bal_auc2, bal_fm2, bal_euc_fm2, ttest_bal_fm2 = eval_cluster(dat + args.qry_attn_test2, model, dat + args.test_pids2,
                                                           dat + args.test_pvecs2, dat + args.test_pids_para2,
                                                           dat + args.test_pvecs_para2, dat + args.test_qids2,
                                                           dat + args.test_qvecs2, args.art_qrels2, args.top_qrels2,
                                                           args.hier_qrels2, args.max_seq)

    print("\nbenchmark Y1 test")
    print("==================")
    print("AUC method all pairs: %.5f (p %.5f), balanced: %.5f (p %.5f)" % (
        all_auc1, ttest_auc1[1], bal_auc1, ttest_bal_auc1[1]))
    print("AUC euclid all pairs: %.5f, balanced: %.5f" % (euc_auc1, bal_euc_auc1))
    print("F1 method all pairs: %.5f (p %.5f), balanced: %.5f (p %.5f)" % (
        all_fm1, ttest_fm1[1], bal_fm1, ttest_bal_fm1[1]))
    print("F1 euclid all pairs: %.5f, balanced: %.5f" % (all_euc_fm1, bal_euc_fm1))
    print("Method top ARI: %.5f (p %.5f), hier ARI: %.5f (p %.5f)" %
          (mean_ari1, paired_ttest_ari1[1], mean_ari_hq1, paired_ttest_ari_hq1[1]))
    print("Euclid top ARI: %.5f, hier ARI: %.5f" % (mean_euc_ari1, mean_euc_ari_hq1))

    print("\nbenchmark Y1 train")
    print("==================")
    print("AUC method all pairs: %.5f (p %.5f), balanced: %.5f (p %.5f)" % (
        all_auc2, ttest_auc2[1], bal_auc2, ttest_bal_auc2[1]))
    print("AUC euclid all pairs: %.5f, balanced: %.5f" % (euc_auc2, bal_euc_auc2))
    print("F1 method all pairs: %.5f (p %.5f), balanced: %.5f (p %.5f)" % (
        all_fm2, ttest_fm2[1], bal_fm2, ttest_bal_fm2[1]))
    print("F1 euclid all pairs: %.5f, balanced: %.5f" % (all_euc_fm2, bal_euc_fm2))
    print("Method top ARI: %.5f (p %.5f), hier ARI: %.5f (p %.5f)" %
          (mean_ari2, paired_ttest_ari2[1], mean_ari_hq2, paired_ttest_ari_hq2[1]))
    print("Euclid top ARI: %.5f, hier ARI: %.5f" % (mean_euc_ari2, mean_euc_ari_hq2))

if __name__ == '__main__':
    main()