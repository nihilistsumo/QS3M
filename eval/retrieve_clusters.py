from model.models import QSSimilarityModel
from data.utils import InputQS3MDatasetBuilder, read_art_qrels
import torch
torch.manual_seed(42)
import numpy as np
from numpy.random import seed
seed(42)
from hashlib import sha1
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from urllib.parse import unquote


def get_clusters(model_path, model_type, qry_attn_file_test, test_pids_file, test_pvecs_file, test_qids_file,
                 test_qvecs_file, article_qrels, top_qrels, hier_qrels):
    model = QSSimilarityModel(768, model_type)
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

    test_data_builder = InputQS3MDatasetBuilder(qry_attn_ts, test_pids, test_pvecs, test_qids, test_qvecs)

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
    clustering_results = {}

    for page in page_paras.keys():
        # print('Going to cluster '+page)
        qid = 'Query: ' +sha1(str.encode(page)).hexdigest()
        if qid not in test_data_builder.query_vecs.keys():
            print(qid + ' not present in query vecs dict')
        else:
            clustering_results[page] = {'query_id': page, 'query_text': unquote(page.split('enwiki:')[1])}

            paralist = page_paras[page]
            true_labels = []
            true_labels_hq = []
            paralist.sort()
            for i in range(len(paralist)):
                true_labels.append(para_labels[paralist[i]])
                true_labels_hq.append(para_labels_hq[paralist[i]])
            X_page, parapairs = test_data_builder.build_cluster_data(qid, paralist)
            pair_scores = model(X_page)
            pair_scores = (pair_scores - torch.min(pair_scores)) / (torch.max(pair_scores) - torch.min(pair_scores))
            pair_score_dict = {}
            pair_baseline_score_dict = {}
            pair_euclid_score_dict = {}
            for i in range(len(parapairs)):
                pair_score_dict[parapairs[i]] = 1 - pair_scores[i].item()
            dist_mat = []
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
                        r.append(pair_score_dict[paralist[i] + '_' + paralist[j]])
                        rbase.append(pair_baseline_score_dict[paralist[i] + '_' + paralist[j]])
                        reuc.append(pair_euclid_score_dict[paralist[i] + '_' + paralist[j]])
                    else:
                        r.append(pair_score_dict[paralist[j] + '_' + paralist[i]])
                        rbase.append(pair_baseline_score_dict[paralist[j] + '_' + paralist[i]])
                        reuc.append(pair_euclid_score_dict[paralist[j] + '_' + paralist[i]])
                dist_mat.append(r)
            clustering_results[page]['elements'] = paralist
            cl = AgglomerativeClustering(n_clusters=page_num_sections[page], affinity='precomputed', linkage='average')
            cl_labels = cl.fit_predict(dist_mat)
            clustering_results[page]['toplevel_cluster_idx'] = cl_labels
            cl_hq = AgglomerativeClustering(n_clusters=page_num_sections_hq[page], affinity='precomputed',
                                            linkage='average')
            cl_labels_hq = cl_hq.fit_predict(dist_mat)
            clustering_results[page]['hier_cluster_idx'] = cl_labels_hq
            ari_score = adjusted_rand_score(true_labels, cl_labels)
            ari_score_hq = adjusted_rand_score(true_labels_hq, cl_labels_hq)
            print(page + ' Method ARI: %.5f' % ari_score)
            pagewise_ari_score[page] = ari_score
            pagewise_hq_ari_score[page] = ari_score_hq

    mean_ari = np.mean(np.array(list(pagewise_ari_score.values())))
    mean_ari_hq = np.mean(np.array(list(pagewise_hq_ari_score.values())))
    return mean_ari, mean_ari_hq, clustering_results
