from sklearn.metrics import roc_auc_score, adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer
from data.utils import read_art_qrels, InputCATSDatasetBuilder
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import json
from hashlib import sha1
import torch

tfidf_vec_dict = {}

def jaccard(p1text, p2text):
    a = set(p1text.split())
    b = set(p2text.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def tfidf_cosine_similarity(pid1, pid2, paratext_dict):
    if pid1 not in tfidf_vec_dict.keys() < 1:
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
    return np.dot(a,b)/(np.sqrt(np.sum(a**2))*np.sqrt(np.sum(b**2)))

def eval_baseline(parapairs_file, test_ptext_file, qry_attn_file_test, test_qids_file, article_qrels, top_qrels):
    all_auc = eval_all_pairs(parapairs_file, test_ptext_file, test_qids_file)
    bal_auc, mean_ari = eval_cluster(test_ptext_file, qry_attn_file_test, test_qids_file, article_qrels, top_qrels)
    print("All AUC: %.5f, Balanced AUC: %.5f, mean ARI: %.5f" %(all_auc, bal_auc, mean_ari))
    tfidf_vec_dict = {}

def eval_all_pairs(parapairs_data, test_ptext_file, test_qids_file):
    ptext_dict = {}
    with open(test_ptext_file, 'r') as f:
        for l in f:
            if len(l.split('\t')) > 1:
                ptext_dict[l.split('\t')[0]] = l.split('\t')[1].strip()
    qry_attn_ts = []
    with open(parapairs_data, 'r') as f:
        parapairs = json.load(f)
    for page in parapairs.keys():
        qid = 'Query:'+sha1(str.encode(page)).hexdigest()
        for i in range(len(parapairs[page]['parapairs'])):
            p1 = parapairs[page]['parapairs'][i].split('_')[0]
            p2 = parapairs[page]['parapairs'][i].split('_')[1]
            qry_attn_ts.append([qid, p1, p2, int(parapairs[page]['labels'][i])])
    test_qids = list(np.load(test_qids_file))
    y = []
    y_score = []
    for qid, pid1, pid2, label in qry_attn_ts:
        if qid in test_qids:
            y.append(float(label))
            #y_score.append(jaccard(ptext_dict[pid1], ptext_dict[pid2]))
            y_score.append(tfidf_cosine_similarity(pid1, pid2, ptext_dict))
    test_auc = roc_auc_score(y, y_score)
    print('\n\nTest all pairs auc: %.5f' % test_auc)
    return test_auc

def eval_cluster(test_ptext_file, qry_attn_file_test, test_qids_file, article_qrels, top_qrels):
    qry_attn_ts = []
    with open(qry_attn_file_test, 'r') as tsf:
        f = True
        for l in tsf:
            if f:
                f = False
                continue
            qry_attn_ts.append(l.split('\t'))
    ptext_dict = {}
    with open(test_ptext_file, 'r') as f:
        for l in f:
            if len(l.split('\t')) > 1:
                ptext_dict[l.split('\t')[0]] = l.split('\t')[1].strip()
    test_qids = list(np.load(test_qids_file))

    y = []
    y_score = []
    for qid, pid1, pid2, label in qry_attn_ts:
        if qid in test_qids:
            y.append(float(label))
            #y_score.append(jaccard(ptext_dict[pid1], ptext_dict[pid2]))
            y_score.append(tfidf_cosine_similarity(pid1, pid2, ptext_dict))
    test_auc = roc_auc_score(y, y_score)
    print('\n\nTest balanced auc: %.5f' % test_auc)

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

    for page in page_paras.keys():
        print('Going to cluster '+page)
        qid = 'Query:'+sha1(str.encode(page)).hexdigest()
        if qid not in test_qids:
            print(qid + ' not present in query vecs dict')
        else:
            paralist = page_paras[page]

            true_labels = []
            for i in range(len(paralist)):
                true_labels.append(para_labels[paralist[i]])
            parapairs = []
            paralist.sort()
            for i in range(len(paralist) - 1):
                for j in range(i + 1, len(paralist)):
                    p1 = paralist[i]
                    p2 = paralist[j]
                    parapairs.append(p1 + '_' + p2)
            pair_scores = np.array([jaccard(ptext_dict[pp.split('_')[0]], ptext_dict[pp.split('_')[1]]) for pp in parapairs])
            pair_scores = (pair_scores - np.min(pair_scores))/(np.max(pair_scores) - np.min(pair_scores))
            pair_score_dict = {}
            for i in range(len(parapairs)):
                pair_score_dict[parapairs[i]] = 1-pair_scores[i]
            dist_mat = []
            paralist.sort()
            for i in range(len(paralist)):
                r = []
                for j in range(len(paralist)):
                    if i == j:
                        r.append(0.0)
                    elif i < j:
                        r.append(pair_score_dict[paralist[i]+ '_' + paralist[j]])
                    else:
                        r.append(pair_score_dict[paralist[j] + '_' + paralist[i]])
                dist_mat.append(r)

            cl = AgglomerativeClustering(n_clusters=page_num_sections[page], affinity='precomputed', linkage='average')
            cl_labels = cl.fit_predict(dist_mat)
            ari_score = adjusted_rand_score(true_labels, cl_labels)
            #print(page+' ARI: %.5f' % ari_score)
            pagewise_ari_score[page] = ari_score
    mean_ari = np.mean(np.array(list(pagewise_ari_score.values())))
    print('Mean ARI score: %.5f' % mean_ari)
    return test_auc, mean_ari

def main():
    eval_baseline('/home/sk1105/sumanta/Mule-data/input_data_v2/pairs/test-cleaned-parapairs/by1-test-cleaned.parapairs.json',
                  '/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/by1test_paratext/by1test_paratext.tsv',
                  '/home/sk1105/sumanta/CATS_data/by1test-qry-attn-bal-allpos-for-eval.tsv',
                  '/home/sk1105/sumanta/CATS_data/by1test-context-qids.npy',
                  '/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-article.qrels',
                  '/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-hierarchical.qrels')

    eval_baseline(
        '/home/sk1105/sumanta/Mule-data/input_data_v2/pairs/train-cleaned-parapairs/by1-train-cleaned.parapairs.json',
        '/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-train-nodup/by1train_paratext/by1train_paratext.tsv',
        '/home/sk1105/sumanta/CATS_data/by1train-qry-attn-bal-allpos.tsv',
        '/home/sk1105/sumanta/CATS_data/by1train-context-qids.npy',
        '/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-article.qrels',
        '/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-train-nodup/train.pages.cbor-hierarchical.qrels')

if __name__ == '__main__':
    main()