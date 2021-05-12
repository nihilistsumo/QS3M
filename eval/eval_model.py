from model.layers import CATS, CATS_Scaled, CATS_QueryScaler, CATS_manhattan
from model.models import CATSSimilarityModel
from model.sent_models import CATSSentenceModel
from data.utils import InputCATSDatasetBuilder, read_art_qrels
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

def eval_all_pairs(parapairs_data, model_path, model_type, test_pids_file, test_pvecs_file, test_qids_file,
                 test_qvecs_file):
    test_pids = np.load(test_pids_file)
    test_pvecs = np.load(test_pvecs_file)
    test_qids = np.load(test_qids_file)
    test_qvecs = np.load(test_qvecs_file)
    model = CATSSimilarityModel(768, model_type)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.cpu()
    with open(parapairs_data, 'r') as f:
        parapairs = json.load(f)
    pagewise_all_auc = {}
    pagewise_all_euc_auc = {}
    pagewise_all_cos_auc = {}
    anchor_auc = []
    cand_auc = []
    anchor_f1 = []
    cand_f1 = []
    qry_attn = []
    for page in parapairs.keys():
        qid = 'Query:' + sha1(str.encode(page)).hexdigest()
        for i in range(len(parapairs[page]['parapairs'])):
            p1 = parapairs[page]['parapairs'][i].split('_')[0]
            p2 = parapairs[page]['parapairs'][i].split('_')[1]
            qry_attn.append([qid, p1, p2, int(parapairs[page]['labels'][i])])

    test_data_builder = InputCATSDatasetBuilder(qry_attn, test_pids, test_pvecs, test_qids, test_qvecs)
    for page in parapairs.keys():
        qry_attn_ts = []
        qid = 'Query:'+sha1(str.encode(page)).hexdigest()
        for i in range(len(parapairs[page]['parapairs'])):
            p1 = parapairs[page]['parapairs'][i].split('_')[0]
            p2 = parapairs[page]['parapairs'][i].split('_')[1]
            qry_attn_ts.append([qid, p1, p2, int(parapairs[page]['labels'][i])])
        X_test, y_test = test_data_builder.build_input_data(qry_attn_ts)
        if len(set(y_test.cpu().numpy())) < 2:
            continue

        ypred_test = model(X_test)
        test_auc = roc_auc_score(y_test.detach().cpu().numpy(), ypred_test.detach().cpu().numpy())
        test_f1 = calc_f1(y_test.detach().cpu().numpy(), ypred_test.detach().cpu().numpy())

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        y_cos = cos(X_test[:, 768:768 * 2], X_test[:, 768 * 2:])
        cos_auc = roc_auc_score(y_test, y_cos)
        cos_f1 = calc_f1(y_test, y_cos)
        y_euclid = torch.sqrt(torch.sum((X_test[:, 768:768 * 2] - X_test[:, 768 * 2:]) ** 2, 1)).numpy()
        y_euclid = 1 - (y_euclid - np.min(y_euclid)) / (np.max(y_euclid) - np.min(y_euclid))
        euclid_auc = roc_auc_score(y_test, y_euclid)
        euclid_f1 = calc_f1(y_test, y_euclid)
        pagewise_all_auc[page] = test_auc
        pagewise_all_euc_auc[page] = euclid_auc
        pagewise_all_cos_auc[page] = cos_auc
        anchor_auc.append(euclid_auc)
        anchor_f1.append(euclid_f1)
        cand_auc.append(test_auc)
        cand_f1.append(test_f1)
        print(page+' Method all-pair AUC: %.5f, F1: %.5f, euclid AUC: %.5f, F1: %.5f, cosine AUC: %.5f, F1: %.5f' %
              (test_auc, test_f1, euclid_auc, euclid_f1, cos_auc, cos_f1))

    paired_ttest = ttest_rel(anchor_auc, cand_auc)
    paired_ttest_f1 = ttest_rel(anchor_f1, cand_f1)
    mean_auc = np.mean(np.array(list(pagewise_all_auc.values())))
    mean_f1 = np.mean(np.array(test_f1))
    mean_euclid_auc = np.mean(np.array(list(pagewise_all_euc_auc.values())))
    mean_euclid_f1 = np.mean(np.array(euclid_f1))
    mean_cos_auc = np.mean(np.array(list(pagewise_all_cos_auc.values())))
    mean_cos_f1 = np.mean(np.array(cos_f1))

    return mean_auc, mean_euclid_auc, mean_cos_auc, paired_ttest, mean_f1, mean_euclid_f1, mean_cos_f1, paired_ttest_f1

def eval_cluster(model_path, model_type, qry_attn_file_test, test_pids_file, test_pvecs_file, test_qids_file,
                 test_qvecs_file, article_qrels, top_qrels, hier_qrels):
    model = CATSSimilarityModel(768, model_type)
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
    #X_test, y_test = test_data_builder.build_input_data()

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

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
    pagewise_base_ari_score = {}
    pagewise_hq_base_ari_score = {}
    pagewise_euc_ari_score = {}
    pagewise_hq_euc_ari_score = {}
    anchor_auc = []
    anchor_f1 = []
    cand_auc = []
    cand_f1 = []
    cos_auc = []
    cos_f1 = []
    anchor_ari_scores = []
    cand_ari_scores = []
    anchor_ari_scores_hq = []
    cand_ari_scores_hq = []

    for page in page_paras.keys():
        #print('Going to cluster '+page)
        qid = 'Query:'+sha1(str.encode(page)).hexdigest()
        if qid not in test_data_builder.query_vecs.keys():
            print(qid + ' not present in query vecs dict')
        else:
            qry_attn_for_page = [d for d in qry_attn_ts if d[0]==qid]
            #test_data_builder_for_page = InputCATSDatasetBuilder(qry_attn_for_page, test_pids, test_pvecs, test_qids, test_qvecs)
            X_test_page, y_test_page = test_data_builder.build_input_data(qry_attn_for_page)
            ypred_test_page = model(X_test_page)
            test_auc_page = roc_auc_score(y_test_page.detach().cpu().numpy(), ypred_test_page.detach().cpu().numpy())
            test_f1_page = calc_f1(y_test_page.detach().cpu().numpy(), ypred_test_page.detach().cpu().numpy())

            y_cos_page = cos(X_test_page[:, 768:768 * 2], X_test_page[:, 768 * 2:])
            cos_auc_page = roc_auc_score(y_test_page, y_cos_page)
            cos_f1_page = calc_f1(y_test_page, y_cos_page)
            y_euclid_page = torch.sqrt(torch.sum((X_test_page[:, 768:768 * 2] - X_test_page[:, 768 * 2:]) ** 2, 1)).numpy()
            y_euclid_page = 1 - (y_euclid_page - np.min(y_euclid_page)) / (np.max(y_euclid_page) - np.min(y_euclid_page))
            euclid_auc_page = roc_auc_score(y_test_page, y_euclid_page)
            euclid_f1_page = calc_f1(y_test_page, y_euclid_page)
            anchor_auc.append(euclid_auc_page)
            anchor_f1.append(euclid_f1_page)
            cand_auc.append(test_auc_page)
            cand_f1.append(test_f1_page)
            cos_auc.append(cos_auc_page)
            cos_f1.append(cos_f1_page)

            paralist = page_paras[page]
            true_labels = []
            true_labels_hq = []
            paralist.sort()
            for i in range(len(paralist)):
                true_labels.append(para_labels[paralist[i]])
                true_labels_hq.append(para_labels_hq[paralist[i]])
            X_page, parapairs = test_data_builder.build_cluster_data(qid, paralist)
            pair_baseline_scores = cos(X_page[:, 768:768 * 2], X_page[:, 768 * 2:])
            pair_euclid_scores = torch.sqrt(torch.sum((X_page[:, 768:768 * 2] - X_page[:, 768 * 2:])**2, 1)).numpy()
            pair_scores = model(X_page)
            pair_scores = (pair_scores - torch.min(pair_scores))/(torch.max(pair_scores) - torch.min(pair_scores))
            pair_baseline_scores = (pair_baseline_scores - torch.min(pair_baseline_scores)) / (torch.max(pair_baseline_scores) - torch.min(pair_baseline_scores))
            pair_euclid_scores = (pair_euclid_scores - np.min(pair_euclid_scores)) / (np.max(pair_euclid_scores) - np.min(pair_euclid_scores))
            pair_score_dict = {}
            pair_baseline_score_dict = {}
            pair_euclid_score_dict = {}
            for i in range(len(parapairs)):
                pair_score_dict[parapairs[i]] = 1-pair_scores[i].item()
                pair_baseline_score_dict[parapairs[i]] = 1-pair_baseline_scores[i]
                pair_euclid_score_dict[parapairs[i]] = pair_euclid_scores[i]
            dist_mat = []
            dist_base_mat = []
            dist_euc_mat = []
            for i in range(len(paralist)):
                r = []
                rbase = []
                reuc = []
                for j in range(len(paralist)):
                    if i == j:
                        r.append(0.0)
                        rbase.append(0.0)
                        reuc.append(0.0)
                    elif i < j:
                        r.append(pair_score_dict[paralist[i]+ '_' + paralist[j]])
                        rbase.append(pair_baseline_score_dict[paralist[i]+ '_' + paralist[j]])
                        reuc.append(pair_euclid_score_dict[paralist[i] + '_' + paralist[j]])
                    else:
                        r.append(pair_score_dict[paralist[j] + '_' + paralist[i]])
                        rbase.append(pair_baseline_score_dict[paralist[j] + '_' + paralist[i]])
                        reuc.append(pair_euclid_score_dict[paralist[j] + '_' + paralist[i]])
                dist_mat.append(r)
                dist_base_mat.append(rbase)
                dist_euc_mat.append(reuc)

            cl = AgglomerativeClustering(n_clusters=page_num_sections[page], affinity='precomputed', linkage='average')
            cl_labels = cl.fit_predict(dist_mat)
            cl_base_labels = cl.fit_predict(dist_base_mat)
            cl_euclid_labels = cl.fit_predict(dist_euc_mat)

            cl_hq = AgglomerativeClustering(n_clusters=page_num_sections_hq[page], affinity='precomputed', linkage='average')
            cl_labels_hq = cl_hq.fit_predict(dist_mat)
            cl_base_labels_hq = cl_hq.fit_predict(dist_base_mat)
            cl_euclid_labels_hq = cl_hq.fit_predict(dist_euc_mat)

            ari_score = adjusted_rand_score(true_labels, cl_labels)
            ari_score_hq = adjusted_rand_score(true_labels_hq, cl_labels_hq)
            ari_base_score = adjusted_rand_score(true_labels, cl_base_labels)
            ari_base_score_hq = adjusted_rand_score(true_labels_hq, cl_base_labels_hq)
            ari_euc_score = adjusted_rand_score(true_labels, cl_euclid_labels)
            ari_euc_score_hq = adjusted_rand_score(true_labels_hq, cl_euclid_labels_hq)
            print(page+' Method bal AUC: %.5f, F1: %.5f, ARI: %.5f, Base bal AUC: %.5f, F1: %.5f, ARI: %.5f, Euclid bal AUC: %.5f, F1: %.5f, ARI: %.5f' %
                  (test_auc_page, test_f1_page, ari_score, cos_auc_page, cos_f1_page, ari_base_score, euclid_auc_page, euclid_f1_page, ari_euc_score))
            pagewise_ari_score[page] = ari_score
            pagewise_base_ari_score[page] = ari_base_score
            pagewise_euc_ari_score[page] = ari_euc_score
            pagewise_hq_ari_score[page] = ari_score_hq
            pagewise_base_ari_score[page] = ari_base_score
            pagewise_hq_base_ari_score[page] = ari_base_score_hq
            pagewise_euc_ari_score[page] = ari_euc_score
            pagewise_hq_euc_ari_score[page] = ari_euc_score_hq
            anchor_ari_scores.append(ari_euc_score)
            cand_ari_scores.append(ari_score)
            anchor_ari_scores_hq.append(ari_euc_score_hq)
            cand_ari_scores_hq.append(ari_score_hq)

    test_auc = np.mean(np.array(cand_auc))
    test_f1 = np.mean(np.array(cand_f1))
    euclid_auc = np.mean(np.array(anchor_auc))
    euclid_f1 = np.mean(np.array(anchor_f1))
    cos_auc = np.mean(np.array(cos_auc))
    cos_f1 = np.mean(np.array(cos_f1))
    paired_ttest_auc = ttest_rel(anchor_auc, cand_auc)
    paired_ttest_f1 = ttest_rel(anchor_f1, cand_f1)
    mean_ari = np.mean(np.array(list(pagewise_ari_score.values())))
    mean_cos_ari = np.mean(np.array(list(pagewise_base_ari_score.values())))
    mean_euc_ari = np.mean(np.array(list(pagewise_euc_ari_score.values())))
    mean_ari_hq = np.mean(np.array(list(pagewise_hq_ari_score.values())))
    mean_cos_ari_hq = np.mean(np.array(list(pagewise_hq_base_ari_score.values())))
    mean_euc_ari_hq = np.mean(np.array(list(pagewise_hq_euc_ari_score.values())))
    '''
    print('Mean ARI score: %.5f' % mean_ari)
    print('Mean Cosine ARI score: %.5f' % mean_cos_ari)
    print('Mean Euclid ARI score: %.5f' % mean_euc_ari)
    
    print('Paired ttest: %.5f, p val: %.5f' % (paired_ttest_ari[0], paired_ttest_ari[1]))
    print('Mean hq ARI score: %.5f' % mean_ari_hq)
    print('Mean hq Cosine ARI score: %.5f' % mean_cos_ari_hq)
    print('Mean hq Euclid ARI score: %.5f' % mean_euc_ari_hq)
    
    print('Paired ttest hq: %.5f, p val: %.5f' % (paired_ttest_ari_hq[0], paired_ttest_ari_hq[1]))
    #with open('/home/sk1105/sumanta/CATS_data/anchor_euc_y1test_hier.json', 'w') as f:
    #    json.dump(pagewise_euc_ari_score, f)
    '''
    paired_ttest_ari = ttest_rel(anchor_ari_scores, cand_ari_scores)
    paired_ttest_ari_hq = ttest_rel(anchor_ari_scores_hq, cand_ari_scores_hq)
    return test_auc, euclid_auc, cos_auc, mean_ari, mean_euc_ari, mean_cos_ari, mean_ari_hq, mean_euc_ari_hq, \
           mean_cos_ari_hq, paired_ttest_ari, paired_ttest_ari_hq, paired_ttest_auc, test_f1, euclid_f1, cos_f1, paired_ttest_f1

def main():

    parser = argparse.ArgumentParser(description='Run CATS model')

    parser.add_argument('-dd', '--data_dir', default="/home/sk1105/sumanta/new_cats_data/")

    parser.add_argument('-qt1', '--qry_attn_test1', default="by1train-qry-attn-bal-allpos.tsv")
    parser.add_argument('-aql1', '--art_qrels1', default="benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-article.qrels")
    parser.add_argument('-tql1', '--top_qrels1', default="benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-toplevel.qrels")
    parser.add_argument('-hql1', '--hier_qrels1', default="benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-hierarchical.qrels")
    parser.add_argument('-pp1', '--parapairs1', default="by1-train-cleaned.parapairs.json")
    parser.add_argument('-tp1', '--test_pids1', default="by1train-all-pids.npy")
    parser.add_argument('-tv1', '--test_pvecs1', default="by1train-all-paravecs.npy")
    parser.add_argument('-tq1', '--test_qids1', default="by1train-context-meanall-qids.npy") #change
    parser.add_argument('-tqv1', '--test_qvecs1', default="by1train-context-meanall-qvecs.npy") #change

    parser.add_argument('-qt2', '--qry_attn_test2', default="by1test-qry-attn-bal-allpos.tsv")
    parser.add_argument('-aql2', '--art_qrels2', default="benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-article.qrels")
    parser.add_argument('-tql2', '--top_qrels2', default="benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-toplevel.qrels")
    parser.add_argument('-hql2', '--hier_qrels2', default="benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-hierarchical.qrels")
    parser.add_argument('-pp2', '--parapairs2', default="by1-test-cleaned.parapairs.json")
    parser.add_argument('-tp2', '--test_pids2', default="by1test-all-pids.npy")
    parser.add_argument('-tv2', '--test_pvecs2', default="by1test-all-paravecs.npy")
    parser.add_argument('-tq2', '--test_qids2', default="by1test-context-meanall-qids.npy") #change
    parser.add_argument('-tqv2', '--test_qvecs2', default="by1test-context-meanall-qvecs.npy") #change

    parser.add_argument('-mt', '--model_type', default="cats") #cats, scaled, abl
    parser.add_argument('-mp', '--model_path', default="/home/sk1105/sumanta/cats_deploy/model/saved_models/cats_meanall_b32_l0.00001_i3.model") #change

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
    parser.add_argument('-mt', '--model_type', default="cats")
    parser.add_argument('-mp', '--model_path',
                        default="/home/sk1105/sumanta/CATS/saved_models/cats_leadpara_b32_l0.00001_i3.model")

    '''
    args = parser.parse_args()
    dat = args.data_dir
    print("\nPagewise benchmark Y1 train")
    print("===========================")
    all_auc1, all_euc_auc1, all_cos_auc1, ttest_auc1, all_fm1, all_euc_fm1, all_cos_fm1, ttest_fm1 = eval_all_pairs(dat + args.parapairs1, args.model_path, args.model_type,
                                                          dat + args.test_pids1, dat + args.test_pvecs1,
                                                          dat + args.test_qids1, dat + args.test_qvecs1)
    bal_auc1, bal_euc_auc1, bal_cos_auc1, mean_ari1, mean_euc_ari1, mean_cos_ari1, mean_ari1_hq, mean_euc_ari1_hq, \
    mean_cos_ari1_hq, ttest1, ttest1_hq, ttest_bal_auc1, bal_fm1, bal_euc_fm1, bal_cos_fm1, ttest_bal_fm1 = eval_cluster(args.model_path,
                                                                                              args.model_type,
                                                                                              dat + args.qry_attn_test1,
                                                                                              dat + args.test_pids1,
                                                                                              dat + args.test_pvecs1,
                                                                                              dat + args.test_qids1,
                                                                                              dat + args.test_qvecs1,
                                                                                              dat + args.art_qrels1,
                                                                                              dat + args.top_qrels1,
                                                                                              dat + args.hier_qrels1)
    print("\nPagewise benchmark Y1 test")
    print("==========================")
    all_auc2, all_euc_auc2, all_cos_auc2, ttest_auc2, all_fm2, all_euc_fm2, all_cos_fm2, ttest_fm2 = eval_all_pairs(dat + args.parapairs2, args.model_path, args.model_type,
                                                          dat + args.test_pids2, dat + args.test_pvecs2,
                                                          dat + args.test_qids2, dat + args.test_qvecs2)
    bal_auc2, bal_euc_auc2, bal_cos_auc2, mean_ari2, mean_euc_ari2, mean_cos_ari2, mean_ari2_hq, mean_euc_ari2_hq, \
    mean_cos_ari2_hq, ttest2, ttest2_hq, ttest_bal_auc2, bal_fm2, bal_euc_fm2, bal_cos_fm2, ttest_bal_fm2 = eval_cluster(args.model_path,
                                                                                              args.model_type,
                                                                                              dat + args.qry_attn_test2,
                                                                                              dat + args.test_pids2,
                                                                                              dat + args.test_pvecs2,
                                                                                              dat + args.test_qids2,
                                                                                              dat + args.test_qvecs2,
                                                                                              dat + args.art_qrels2,
                                                                                              dat + args.top_qrels2,
                                                                                              dat + args.hier_qrels2)
    print("\nbenchmark Y1 test")
    print("==================")
    print("AUC method all pairs: %.5f (p %.5f), balanced: %.5f (p %.5f)" % (
    all_auc2, ttest_auc2[1], bal_auc2, ttest_bal_auc2[1]))
    print("AUC euclid all pairs: %.5f, balanced: %.5f" % (all_euc_auc2, bal_euc_auc2))
    print("AUC cosine all pairs: %.5f, balanced: %.5f" % (all_cos_auc2, bal_cos_auc2))
    print("F1 method all pairs: %.5f (p %.5f), balanced: %.5f (p %.5f)" % (
        all_fm2, ttest_fm2[1], bal_fm2, ttest_bal_fm2[1]))
    print("F1 euclid all pairs: %.5f, balanced: %.5f" % (all_euc_fm2, bal_euc_fm2))
    print("F1 cosine all pairs: %.5f, balanced: %.5f" % (all_cos_fm2, bal_cos_fm2))
    print("Method top ARI: %.5f (p %.5f), hier ARI: %.5f (p %.5f)" %
          (mean_ari2, ttest2[1], mean_ari2_hq, ttest2_hq[1]))
    print("Euclid top ARI: %.5f, hier ARI: %.5f" % (mean_euc_ari2, mean_euc_ari2_hq))
    print("Cosine top ARI: %.5f, hier ARI: %.5f" % (mean_cos_ari2, mean_cos_ari2_hq))

    print("\nbenchmark Y1 train")
    print("==================")
    print("AUC method all pairs: %.5f (p %.5f), balanced: %.5f (p %.5f)" % (
        all_auc1, ttest_auc1[1], bal_auc1, ttest_bal_auc1[1]))
    print("AUC euclid all pairs: %.5f, balanced: %.5f" % (all_euc_auc1, bal_euc_auc1))
    print("AUC cosine all pairs: %.5f, balanced: %.5f" % (all_cos_auc1, bal_cos_auc1))
    print("F1 method all pairs: %.5f (p %.5f), balanced: %.5f (p %.5f)" % (
        all_fm1, ttest_fm1[1], bal_fm1, ttest_bal_fm1[1]))
    print("F1 euclid all pairs: %.5f, balanced: %.5f" % (all_euc_fm1, bal_euc_fm1))
    print("F1 cosine all pairs: %.5f, balanced: %.5f" % (all_cos_fm1, bal_cos_fm1))
    print("Method top ARI: %.5f (p %.5f), hier ARI: %.5f (p %.5f)" % (mean_ari1, ttest1[1], mean_ari1_hq, ttest1_hq[1]))
    print("Euclid top ARI: %.5f, hier ARI: %.5f" % (mean_euc_ari1, mean_euc_ari1_hq))
    print("Cosine top ARI: %.5f, hier ARI: %.5f" % (mean_cos_ari1, mean_cos_ari1_hq))


if __name__ == '__main__':
    main()