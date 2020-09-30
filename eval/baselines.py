from sklearn.metrics import roc_auc_score, adjusted_rand_score
from data.utils import read_art_qrels
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import json

def jaccard(p1text, p2text):
    a = set(p1text.split())
    b = set(p2text.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

def eval_baseline(parapairs_file, test_ptext_file, qry_attn_file_test, article_qrels, top_qrels):
    with open(parapairs_file, 'r') as f:
        parapairs_data = json.load(f)
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
    y_true = [d[3] for d in qry_attn_ts]
    y_pred = [jaccard(ptext_dict[d[1]], ptext_dict[d[2]]) for d in qry_attn_ts]
    bal_auc = roc_auc_score(y_true, y_pred)

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

    page_auc = []
    for page in page_paras.keys():
        print('Going to cluster ' + page)
        paralist = page_paras[page]

        true_labels = []
        for i in range(len(paralist)):
            true_labels.append(para_labels[paralist[i]])
        parapairs = []
        labels = []
        '''
        for i in range(len(paralist)-1):
            for j in range(i+1, len(paralist)):
                p1 = paralist[i]
                p2 = paralist[j]
                parapairs.append(p1+'_'+p2)
        '''
        for i in range(len(parapairs_data[page]['parapairs'])):
            parapairs.append(parapairs_data[page]['parapairs'][i])
            labels.append(parapairs_data[page]['labels'][i])
        pair_scores = [jaccard(ptext_dict[parapairs[i].split('_')[0]], ptext_dict[parapairs[i].split('_')[1]])
                       for i in range(len(parapairs))]
        #true_labels = [parapairs_data[page]['labels'][i] for i in range(len(parapairs))]
        true_labels = []
        for i in range(len(paralist)):
            true_labels.append(para_labels[paralist[i]])
        page_auc.append(roc_auc_score(labels, pair_scores))
        pair_scores = [(p - min(pair_scores)) / (max(pair_scores) - min(pair_scores)) for p in pair_scores]
        pair_score_dict = {}
        for i in range(len(parapairs)):
            pair_score_dict[parapairs[i]] = 1 - pair_scores[i]
        dist_mat = []
        paralist.sort()
        for i in range(len(paralist)):
            r = []
            for j in range(len(paralist)):
                if i == j:
                    r.append(0.0)
                elif i < j:
                    r.append(pair_score_dict[paralist[i] + '_' + paralist[j]])
                else:
                    r.append(pair_score_dict[paralist[j] + '_' + paralist[i]])
            dist_mat.append(r)

        cl = AgglomerativeClustering(n_clusters=page_num_sections[page], affinity='precomputed', linkage='average')
        cl_labels = cl.fit_predict(dist_mat)
        ari_score = adjusted_rand_score(true_labels, cl_labels)
        print(page + ' ARI: %.5f' % ari_score)
        pagewise_ari_score[page] = ari_score
    print('All pairs AUC: %.5f' % np.mean(np.array(page_auc)))
    print('Balanced AUC: %.5f' % bal_auc)
    print('Mean ARI score: %.5f' % np.mean(np.array(list(pagewise_ari_score.values()))))

def main():
    eval_baseline('/home/sk1105/sumanta/Mule-data/input_data_v2/pairs/test-cleaned-parapairs/by1-test-cleaned.parapairs.json',
                  '/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/by1test_paratext/by1test_paratext.tsv',
                  '/home/sk1105/sumanta/CATS_data/by1test-qry-attn-bal-allpos-for-eval.tsv',
                  '/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-article.qrels',
                  '/home/sk1105/sumanta/trec_dataset/benchmarkY1/benchmarkY1-test-nodup/test.pages.cbor-toplevel.qrels')

if __name__ == '__main__':
    main()