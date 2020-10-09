from sklearn.metrics import roc_auc_score, adjusted_rand_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from data.utils import read_art_qrels, InputCATSDatasetBuilder
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import json
from hashlib import sha1
import math
import torch
from scipy.stats import ttest_rel
from scipy.special import kl_div
import argparse
from nltk.corpus import stopwords
from collections import defaultdict
from gensim import corpora
from gensim.models import ldamodel

tfidf_vec_dict = {}
lda_tm_topic_dist = {}
num_topics=200 #for topic model

def lda_topic_model(test_ptext_path, train_token_dict_path, trained_model_path):
    ptext_dict = {}
    with open(test_ptext_path, 'r') as f:
        for l in f:
            if len(l.split('\t')) > 1:
                ptext_dict[l.split('\t')[0]] = l.split('\t')[1].strip()
    model = ldamodel.LdaModel.load(trained_model_path)
    token_dict = corpora.Dictionary.load(train_token_dict_path)
    stops = stopwords.words('english')
    paraids = list(ptext_dict.keys())
    raw_docs = [ptext_dict[k] for k in paraids]
    pre_docs = [[word for word in doc.lower().split() if word not in stops] for doc in raw_docs]
    frequency = defaultdict(int)
    for d in pre_docs:
        for t in d:
            frequency[t] += 1
    texts = [[t for t in doc if frequency[t] > 1] for doc in pre_docs]
    unseen_corpus = [token_dict.doc2bow(text) for text in texts]
    for p in range(len(paraids)):
        topic_vec = model[unseen_corpus[p]]
        lda_tm_topic_dist[paraids[p]] = [(t[0], float(t[1])) for t in topic_vec]

def calc_f1(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    yp = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
    yp = np.array([1.0 if d > 0.5 else 0.0 for d in yp])
    test_f1 = f1_score(y_true, yp)
    return test_f1

def jaccard(p1text, p2text):
    a = set(p1text.split())
    b = set(p2text.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def kldiv(a, b):
    score = 0
    for s in kl_div(a, b):
        if s != float("inf"):
            score += s
    return score

def sparse_jsdiv_score(p1, p2):
    v1 = lda_tm_topic_dist[p1]
    v2 = lda_tm_topic_dist[p2]
    x = [0] * num_topics
    for v in v1:
        x[v[0]] = v[1]
    y = [0] * num_topics
    for v in v2:
        y[v[0]] = v[1]
    m = [(x[i]+y[i])/2 for i in range(num_topics)]
    kldiv1 = kldiv(x, m)
    kldiv2 = kldiv(y, m)
    return (kldiv1 + kldiv2)/2

def tfidf_cosine_similarity(pid1, pid2, paratext_dict):
    if pid1 not in tfidf_vec_dict.keys():
        pid_list = list(paratext_dict.keys())
        corpus = []
        for i in range(len(pid_list)):
            corpus.append(paratext_dict[pid_list[i]].strip())
        tfidf = TfidfVectorizer()
        vecs = tfidf.fit_transform(corpus).toarray()
        for i in range(len(pid_list)):
            tfidf_vec_dict[pid_list[i]] = vecs[i]
    a = tfidf_vec_dict[pid1]
    b = tfidf_vec_dict[pid2]
    score = np.dot(a,b)/(np.sqrt(np.sum(a**2))*np.sqrt(np.sum(b**2)))
    if math.isnan(score):
        return 0.0
    else:
        return score

def eval_all_pairs(parapairs_data, test_ptext_file, test_pids_file, test_pvecs_file, test_qids_file, test_qvecs_file):
    ptext_dict = {}
    with open(test_ptext_file, 'r') as f:
        for l in f:
            if len(l.split('\t')) > 1:
                ptext_dict[l.split('\t')[0]] = l.split('\t')[1].strip()

    test_pids = np.load(test_pids_file)
    test_pvecs = np.load(test_pvecs_file)
    test_qids = np.load(test_qids_file)
    test_qvecs = np.load(test_qvecs_file)
    with open(parapairs_data, 'r') as f:
        parapairs = json.load(f)
    anchor_auc = []
    anchor_f1 = []
    cand_auc = []
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
        qid = 'Query:' + sha1(str.encode(page)).hexdigest()
        y = []
        y_baseline = []
        for i in range(len(parapairs[page]['parapairs'])):
            p1 = parapairs[page]['parapairs'][i].split('_')[0]
            p2 = parapairs[page]['parapairs'][i].split('_')[1]
            qry_attn_ts.append([qid, p1, p2, int(parapairs[page]['labels'][i])])
            y.append(int(parapairs[page]['labels'][i]))
            #y_baseline.append(tfidf_cosine_similarity(p1, p2, ptext_dict))
            #y_baseline.append(jaccard(ptext_dict[p1], ptext_dict[p2]))
            y_baseline.append(sparse_jsdiv_score(p1, p2))
        X_test, y_test = test_data_builder.build_input_data(qry_attn_ts)
        if len(set(y_test.cpu().numpy())) < 2:
            continue

        method_auc = roc_auc_score(y, y_baseline)
        method_f1 = calc_f1(y, y_baseline)
        y_euclid = torch.sqrt(torch.sum((X_test[:, 768:768 * 2] - X_test[:, 768 * 2:]) ** 2, 1)).numpy()
        y_euclid = 1 - (y_euclid - np.min(y_euclid)) / (np.max(y_euclid) - np.min(y_euclid))
        euclid_auc = roc_auc_score(y_test, y_euclid)
        euclid_f1 = calc_f1(y_test, y_euclid)
        cand_auc.append(method_auc)
        cand_f1.append(method_f1)
        anchor_auc.append(euclid_auc)
        anchor_f1.append(euclid_f1)
        print(page + ' Method all-pair AUC: %.5f, F1: %.5f, euclid AUC: %.5f, F1: %.5f' % (method_auc, method_f1, euclid_auc, euclid_f1))

    paired_ttest = ttest_rel(anchor_auc, cand_auc)
    paired_ttest_f1 = ttest_rel(anchor_f1, cand_f1)
    mean_auc = np.mean(np.array(cand_auc))
    mean_f1 = np.mean(np.array(cand_f1))
    mean_euclid_auc = np.mean(np.array(anchor_auc))
    mean_euclid_f1 = np.mean(np.array(anchor_f1))

    return mean_auc, mean_euclid_auc, paired_ttest, mean_f1, mean_euclid_f1, paired_ttest_f1

def eval_cluster(qry_attn_file_test, test_ptext_file, test_pids_file, test_pvecs_file, test_qids_file,
                 test_qvecs_file, article_qrels, top_qrels, hier_qrels):
    ptext_dict = {}
    with open(test_ptext_file, 'r') as f:
        for l in f:
            if len(l.split('\t')) > 1:
                ptext_dict[l.split('\t')[0]] = l.split('\t')[1].strip()
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

    anchor_auc = []
    cand_auc = []
    anchor_f1 = []
    cand_f1 = []
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
            X_test_page, y_test_page, page_pairs = test_data_builder.build_input_data_with_pairs(qry_attn_for_page)
            #pair_scores_bal = [tfidf_cosine_similarity(pp.split('_')[0], pp.split('_')[1], ptext_dict) for pp in page_pairs]
            #pair_scores_bal = [jaccard(ptext_dict[pp.split('_')[0]], ptext_dict[pp.split('_')[1]]) for pp in page_pairs]
            pair_scores_bal = [sparse_jsdiv_score(pp.split('_')[0], pp.split('_')[1]) for pp in page_pairs]
            pair_scores_bal = (pair_scores_bal - np.min(pair_scores_bal)) / (np.max(pair_scores_bal) - np.min(pair_scores_bal))
            test_auc_page = roc_auc_score(y_test_page, pair_scores_bal)
            cand_auc.append(test_auc_page)
            test_f1_page = calc_f1(y_test_page, pair_scores_bal)
            cand_f1.append(test_f1_page)

            y_euclid_page = torch.sqrt(torch.sum((X_test_page[:, 768:768 * 2] - X_test_page[:, 768 * 2:]) ** 2, 1)).numpy()
            y_euclid_page = 1 - (y_euclid_page - np.min(y_euclid_page)) / (np.max(y_euclid_page) - np.min(y_euclid_page))
            euclid_auc_page = roc_auc_score(y_test_page, y_euclid_page)
            anchor_auc.append(euclid_auc_page)
            euclid_f1_page = calc_f1(y_test_page, y_euclid_page)
            anchor_f1.append(euclid_f1_page)

            paralist = page_paras[page]
            true_labels = []
            true_labels_hq = []
            paralist.sort()
            for i in range(len(paralist)):
                true_labels.append(para_labels[paralist[i]])
                true_labels_hq.append(para_labels_hq[paralist[i]])
            X_page, parapairs = test_data_builder.build_cluster_data(qid, paralist)
            #pair_scores = [tfidf_cosine_similarity(pp.split('_')[0], pp.split('_')[1], ptext_dict) for pp in parapairs]
            #pair_scores = [jaccard(ptext_dict[pp.split('_')[0]], ptext_dict[pp.split('_')[1]]) for pp in parapairs]
            pair_scores = [sparse_jsdiv_score(pp.split('_')[0], pp.split('_')[1]) for pp in parapairs]
            pair_scores = (pair_scores - np.min(pair_scores)) / (np.max(pair_scores) - np.min(pair_scores))
            pair_euclid_scores = torch.sqrt(torch.sum((X_page[:, 768:768 * 2] - X_page[:, 768 * 2:])**2, 1)).numpy()
            pair_euclid_scores = (pair_euclid_scores - np.min(pair_euclid_scores)) / (np.max(pair_euclid_scores) - np.min(pair_euclid_scores))
            pair_score_dict = {}
            pair_euclid_score_dict = {}
            for i in range(len(parapairs)):
                pair_score_dict[parapairs[i]] = 1 - pair_scores[i]
                pair_euclid_score_dict[parapairs[i]] = pair_euclid_scores[i]
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
            anchor_ari_scores.append(ari_euc_score)
            cand_ari_scores.append(ari_score)
            anchor_ari_scores_hq.append(ari_euc_score_hq)
            cand_ari_scores_hq.append(ari_score_hq)

    test_auc = np.mean(np.array(cand_auc))
    euclid_auc = np.mean(np.array(anchor_auc))
    paired_ttest_auc = ttest_rel(anchor_auc, cand_auc)
    test_f1 = np.mean(np.array(cand_f1))
    euclid_f1 = np.mean(np.array(anchor_f1))
    paired_ttest_f1 = ttest_rel(anchor_f1, cand_f1)
    mean_ari = np.mean(np.array(cand_ari_scores))
    mean_euc_ari = np.mean(np.array(anchor_ari_scores))
    mean_ari_hq = np.mean(np.array(cand_ari_scores_hq))
    mean_euc_ari_hq = np.mean(np.array(anchor_ari_scores_hq))
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

    parser.add_argument('-dd', '--data_dir', default="/home/sk1105/sumanta/CATS_data/")
    parser.add_argument('-tm', '--topic_model', default="/home/sk1105/sumanta/CATS_data/topic_model/topic_model_half-y1train-qry-attn-t200.model")
    parser.add_argument('-td', '--token_dict', default="/home/sk1105/sumanta/CATS_data/topic_model/half-y1train-qry-attn-lda-tm-t200.tokendict")

    parser.add_argument('-qt1', '--qry_attn_test1', default="by1train-qry-attn-bal-allpos.tsv")
    parser.add_argument('-aql1', '--art_qrels1', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-article.qrels")
    parser.add_argument('-tql1', '--top_qrels1', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-toplevel.qrels")
    parser.add_argument('-hql1', '--hier_qrels1', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-hierarchical.qrels")
    parser.add_argument('-pp1', '--parapairs1', default="/home/sk1105/sumanta/Mule-data/input_data_v2/pairs/train-cleaned-parapairs/by1-train-cleaned.parapairs.json")
    parser.add_argument('-ptx1', '--ptext_file1', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-train-nodup/by1train_paratext/by1train_paratext.tsv")
    parser.add_argument('-tp1', '--test_pids1', default="by1train-all-pids.npy")
    parser.add_argument('-tv1', '--test_pvecs1', default="by1train-all-paravecs.npy")
    parser.add_argument('-tq1', '--test_qids1', default="by1train-context-qids.npy")
    parser.add_argument('-tqv1', '--test_qvecs1', default="by1train-context-qvecs.npy")

    parser.add_argument('-qt2', '--qry_attn_test2', default="by1test-qry-attn-bal-allpos-for-eval.tsv")
    parser.add_argument('-aql2', '--art_qrels2', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-article.qrels")
    parser.add_argument('-tql2', '--top_qrels2', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-toplevel.qrels")
    parser.add_argument('-hql2', '--hier_qrels2', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-hierarchical.qrels")
    parser.add_argument('-pp2', '--parapairs2', default="/home/sk1105/sumanta/Mule-data/input_data_v2/pairs/test-cleaned-parapairs/by1-test-cleaned.parapairs.json")
    parser.add_argument('-ptx2', '--ptext_file2', default="/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/by1test_paratext/by1test_paratext.tsv")
    parser.add_argument('-tp2', '--test_pids2', default="by1test-all-pids.npy")
    parser.add_argument('-tv2', '--test_pvecs2', default="by1test-all-paravecs.npy")
    parser.add_argument('-tq2', '--test_qids2', default="by1test-context-qids.npy")
    parser.add_argument('-tqv2', '--test_qvecs2', default="by1test-context-qvecs.npy")

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
    ##### for topic model#####
    lda_topic_model(args.ptext_file1, args.token_dict, args.topic_model)
    ##########################
    print("\nPagewise benchmark Y1 train")
    print("===========================")
    all_auc1, all_euc_auc1, ttest_auc1, all_fm1, all_euc_fm1, ttest_fm1 = eval_all_pairs(args.parapairs1, args.ptext_file1,
                                                                      dat + args.test_pids1, dat + args.test_pvecs1,
                                                                      dat + args.test_qids1, dat + args.test_qvecs1)

    bal_auc1, bal_euc_auc1, mean_ari1, mean_euc_ari1, mean_ari1_hq, mean_euc_ari1_hq, \
    ttest1, ttest1_hq, ttest_bal_auc1, bal_fm1, bal_euc_fm1, ttest_bal_fm1 = eval_cluster(dat + args.qry_attn_test1,
                                                                       args.ptext_file1,
                                                                       dat + args.test_pids1,
                                                                       dat + args.test_pvecs1,
                                                                       dat + args.test_qids1,
                                                                       dat + args.test_qvecs1,
                                                                       args.art_qrels1,
                                                                       args.top_qrels1,
                                                                       args.hier_qrels1)
    ##### for topic model#####
    lda_topic_model(args.ptext_file2, args.token_dict, args.topic_model)
    ##########################
    print("\nPagewise benchmark Y1 test")
    print("==========================")
    all_auc2, all_euc_auc2, ttest_auc2, all_fm2, all_euc_fm2, ttest_fm2 = eval_all_pairs(args.parapairs2, args.ptext_file2,
                                                        dat + args.test_pids2, dat + args.test_pvecs2,
                                                        dat + args.test_qids2, dat + args.test_qvecs2)

    bal_auc2, bal_euc_auc2, mean_ari2, mean_euc_ari2, mean_ari2_hq, mean_euc_ari2_hq, \
    ttest2, ttest2_hq, ttest_bal_auc2, bal_fm2, bal_euc_fm2, ttest_bal_fm2 = eval_cluster(dat + args.qry_attn_test2,
                                                     args.ptext_file2,
                                                     dat + args.test_pids2,
                                                     dat + args.test_pvecs2,
                                                     dat + args.test_qids2,
                                                     dat + args.test_qvecs2,
                                                     args.art_qrels2,
                                                     args.top_qrels2,
                                                     args.hier_qrels2)
    print("\nbenchmark Y1 test")
    print("==================")
    print("AUC method all pairs: %.5f (p %.5f), balanced: %.5f (p %.5f)" % (
        all_auc2, ttest_auc2[1], bal_auc2, ttest_bal_auc2[1]))
    print("AUC euclid all pairs: %.5f, balanced: %.5f" % (all_euc_auc2, bal_euc_auc2))
    print("F1 method all pairs: %.5f (p %.5f), balanced: %.5f (p %.5f)" % (
        all_fm2, ttest_fm2[1], bal_fm2, ttest_bal_fm2[1]))
    print("F1 euclid all pairs: %.5f, balanced: %.5f" % (all_euc_fm2, bal_euc_fm2))
    print("Method top ARI: %.5f (p %.5f), hier ARI: %.5f (p %.5f)" %
          (mean_ari2, ttest2[1], mean_ari2_hq, ttest2_hq[1]))
    print("Euclid top ARI: %.5f, hier ARI: %.5f" % (mean_euc_ari2, mean_euc_ari2_hq))

    print("\nbenchmark Y1 train")
    print("==================")
    print("AUC method all pairs: %.5f (p %.5f), balanced: %.5f (p %.5f)" % (
        all_auc1, ttest_auc1[1], bal_auc1, ttest_bal_auc1[1]))
    print("AUC euclid all pairs: %.5f, balanced: %.5f" % (all_euc_auc1, bal_euc_auc1))
    print("F1 method all pairs: %.5f (p %.5f), balanced: %.5f (p %.5f)" % (
        all_fm1, ttest_fm1[1], bal_fm1, ttest_bal_fm1[1]))
    print("F1 euclid all pairs: %.5f, balanced: %.5f" % (all_euc_fm1, bal_euc_fm1))
    print("Method top ARI: %.5f (p %.5f), hier ARI: %.5f (p %.5f)" % (
        mean_ari1, ttest1[1], mean_ari1_hq, ttest1_hq[1]))
    print("Euclid top ARI: %.5f, hier ARI: %.5f" % (mean_euc_ari1, mean_euc_ari1_hq))


if __name__ == '__main__':
    main()