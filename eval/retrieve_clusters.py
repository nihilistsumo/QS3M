from model.models import QSSimilarityModel
from data.utils import InputQS3MDatasetBuilder, read_art_qrels
import torch
torch.manual_seed(42)
import torch.nn as nn
import numpy as np
from numpy.random import seed
seed(42)
from hashlib import sha1
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from urllib.parse import unquote
from sentence_transformers import SentenceTransformer
import json


def build_cluster_data(query_vec, paralist, all_para_vec_dict):
    X = []
    parapairs = []
    paralist.sort()
    for i in range(len(paralist) - 1):
        for j in range(i + 1, len(paralist)):
            p1 = paralist[i]
            p2 = paralist[j]
            row = np.hstack((query_vec, all_para_vec_dict[p1], all_para_vec_dict[p2]))
            X.append(row)
            parapairs.append(p1 + '_' + p2)
    X = torch.tensor(X)
    return X, parapairs


def get_baseline_clusters(qry_attn_file_test, test_pids_file, test_pvecs_file, test_qids_file, test_qvecs_file,
                          article_qrels, top_qrels, hier_qrels):
    test_pids = np.load(test_pids_file)
    test_pvecs = np.load(test_pvecs_file)
    test_qids = np.load(test_qids_file)
    test_qvecs = np.load(test_qvecs_file)
    qry_attn_ts = []
    with open(qry_attn_file_test, 'r') as tsf:
        f = True
        for l in tsf:
            if f:
                f = False
                continue
            qry_attn_ts.append(l.split('\t'))

    test_data_builder = InputQS3MDatasetBuilder(qry_attn_ts, test_pids, test_pvecs, test_qids, test_qvecs)

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

    pagewise_base_ari_score = {}
    pagewise_hq_base_ari_score = {}
    pagewise_euc_ari_score = {}
    pagewise_hq_euc_ari_score = {}
    clustering_results_base = {}
    clustering_results_euc = {}

    for page in page_paras.keys():
        qid = 'Query:' + sha1(str.encode(page)).hexdigest()
        if qid not in test_data_builder.query_vecs.keys():
            print(qid + ' not present in query vecs dict')
        else:
            clustering_results_base[page] = {'query_id': page, 'query_text': unquote(page.split('enwiki:')[1])}
            clustering_results_euc[page] = {'query_id': page, 'query_text': unquote(page.split('enwiki:')[1])}
            paralist = page_paras[page]
            true_labels = []
            true_labels_hq = []
            paralist.sort()
            for i in range(len(paralist)):
                true_labels.append(para_labels[paralist[i]])
                true_labels_hq.append(para_labels_hq[paralist[i]])
            X_page, parapairs = test_data_builder.build_cluster_data(qid, paralist)
            pair_baseline_scores = cos(X_page[:, 768:768 * 2], X_page[:, 768 * 2:])
            pair_euclid_scores = torch.sqrt(torch.sum((X_page[:, 768:768 * 2] - X_page[:, 768 * 2:]) ** 2, 1)).numpy()
            pair_baseline_scores = (pair_baseline_scores - torch.min(pair_baseline_scores)) / (
                    torch.max(pair_baseline_scores) - torch.min(pair_baseline_scores))
            pair_euclid_scores = (pair_euclid_scores - np.min(pair_euclid_scores)) / (
                    np.max(pair_euclid_scores) - np.min(pair_euclid_scores))
            pair_baseline_score_dict = {}
            pair_euclid_score_dict = {}
            for i in range(len(parapairs)):
                pair_baseline_score_dict[parapairs[i]] = 1 - pair_baseline_scores[i]
                pair_euclid_score_dict[parapairs[i]] = pair_euclid_scores[i]
            dist_base_mat = []
            dist_euc_mat = []
            for i in range(len(paralist)):
                rbase = []
                reuc = []
                for j in range(len(paralist)):
                    if i == j:
                        rbase.append(0.0)
                        reuc.append(0.0)
                    elif i < j:
                        rbase.append(pair_baseline_score_dict[paralist[i] + '_' + paralist[j]])
                        reuc.append(pair_euclid_score_dict[paralist[i] + '_' + paralist[j]])
                    else:
                        rbase.append(pair_baseline_score_dict[paralist[j] + '_' + paralist[i]])
                        reuc.append(pair_euclid_score_dict[paralist[j] + '_' + paralist[i]])
                dist_base_mat.append(rbase)
                dist_euc_mat.append(reuc)

            cl = AgglomerativeClustering(n_clusters=page_num_sections[page], affinity='precomputed', linkage='average')
            clustering_results_base[page]['elements'] = paralist
            clustering_results_euc[page]['elements'] = paralist
            cl_base_labels = cl.fit_predict(dist_base_mat).tolist()
            cl_euclid_labels = cl.fit_predict(dist_euc_mat).tolist()
            clustering_results_base[page]['toplevel_clustering_idx'] = cl_base_labels
            clustering_results_euc[page]['toplevel_clustering_idx'] = cl_euclid_labels

            cl_hq = AgglomerativeClustering(n_clusters=page_num_sections_hq[page], affinity='precomputed',
                                            linkage='average')
            cl_base_labels_hq = cl_hq.fit_predict(dist_base_mat).tolist()
            cl_euclid_labels_hq = cl_hq.fit_predict(dist_euc_mat).tolist()
            clustering_results_base[page]['hier_clustering_idx'] = cl_base_labels_hq
            clustering_results_euc[page]['hier_clustering_idx'] = cl_euclid_labels_hq

            ari_base_score = adjusted_rand_score(true_labels, cl_base_labels)
            ari_base_score_hq = adjusted_rand_score(true_labels_hq, cl_base_labels_hq)
            ari_euc_score = adjusted_rand_score(true_labels, cl_euclid_labels)
            ari_euc_score_hq = adjusted_rand_score(true_labels_hq, cl_euclid_labels_hq)
            print(page + ' Base ARI: %.5f, Euclid ARI: %.5f' % (ari_base_score, ari_euc_score))
            pagewise_base_ari_score[page] = ari_base_score
            pagewise_euc_ari_score[page] = ari_euc_score
            pagewise_base_ari_score[page] = ari_base_score
            pagewise_hq_base_ari_score[page] = ari_base_score_hq
            pagewise_euc_ari_score[page] = ari_euc_score
            pagewise_hq_euc_ari_score[page] = ari_euc_score_hq

    mean_cos_ari = np.mean(np.array(list(pagewise_base_ari_score.values())))
    mean_euc_ari = np.mean(np.array(list(pagewise_euc_ari_score.values())))
    mean_cos_ari_hq = np.mean(np.array(list(pagewise_hq_base_ari_score.values())))
    mean_euc_ari_hq = np.mean(np.array(list(pagewise_hq_euc_ari_score.values())))
    return mean_cos_ari, mean_cos_ari_hq, clustering_results_base, mean_euc_ari, mean_euc_ari_hq, clustering_results_euc


def get_clusters(model_path, model_type, test_pids_file, test_pvecs_file, test_qids_file,
                 test_qvecs_file, article_qrels, top_qrels, hier_qrels):
    model = QSSimilarityModel(768, model_type)
    model.load_state_dict(torch.load(model_path))
    model.eval()
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
        qid = 'Query:' + sha1(str.encode(page)).hexdigest()
        clustering_results[page] = {'query_id': page, 'query_text': unquote(page.split('enwiki:')[1])}

        paralist = page_paras[page]
        true_labels = []
        true_labels_hq = []
        paralist.sort()
        for i in range(len(paralist)):
            true_labels.append(para_labels[paralist[i]])
            true_labels_hq.append(para_labels_hq[paralist[i]])
        X_page, parapairs = build_cluster_data(qid, paralist, test_qids, test_qvecs, test_pids, test_pvecs)
        pair_scores = model(X_page)
        pair_scores = (pair_scores - torch.min(pair_scores)) / (torch.max(pair_scores) - torch.min(pair_scores))
        pair_score_dict = {}
        for i in range(len(parapairs)):
            pair_score_dict[parapairs[i]] = 1 - pair_scores[i].item()
        dist_mat = []
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
        clustering_results[page]['elements'] = paralist
        cl = AgglomerativeClustering(n_clusters=page_num_sections[page], affinity='precomputed', linkage='average')
        cl_labels = cl.fit_predict(dist_mat).tolist()
        clustering_results[page]['toplevel_cluster_idx'] = cl_labels
        cl_hq = AgglomerativeClustering(n_clusters=page_num_sections_hq[page], affinity='precomputed',
                                        linkage='average')
        cl_labels_hq = cl_hq.fit_predict(dist_mat).tolist()
        clustering_results[page]['hier_cluster_idx'] = cl_labels_hq
        ari_score = adjusted_rand_score(true_labels, cl_labels)
        ari_score_hq = adjusted_rand_score(true_labels_hq, cl_labels_hq)
        print(page + ' Method ARI: %.5f' % ari_score)
        pagewise_ari_score[page] = ari_score
        pagewise_hq_ari_score[page] = ari_score_hq

    mean_ari = np.mean(np.array(list(pagewise_ari_score.values())))
    mean_ari_hq = np.mean(np.array(list(pagewise_hq_ari_score.values())))
    return mean_ari, mean_ari_hq, clustering_results


def get_clusters_from_run(model_path, model_type, paratext_file, query_context_file, cand_run_qrels_format, top_qrels,
                          hier_qrels, emb_model_name='bert-base-uncased', query_type='title'):
    model = QSSimilarityModel(768, model_type)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    emb_model = SentenceTransformer(emb_model_name)
    paratexts = []
    paraids = []
    with open(paratext_file, 'r', encoding='utf8') as f:
        for l in f:
            paraids.append(l.split('\t')[0])
            paratexts.append(l.split('\t')[1].strip())
    paravecs = emb_model.encode(paratexts)
    paravec_dict = {}
    for i, para in enumerate(paraids):
        paravec_dict[para] = paravecs[i]
    with open(query_context_file, 'r') as f:
        query_context = json.load(f)

    page_paras = read_art_qrels(cand_run_qrels_format)
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
            if p in para_labels.keys():
                sec.add(para_labels[p])
            else:
                sec.add('nonrel')
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
            if p in para_labels_hq.keys():
                sec.add(para_labels_hq[p])
            else:
                sec.add('nonrel')
        page_num_sections_hq[page] = len(sec)

    clustering_results = {}

    for page in page_paras.keys():
        qid = 'Query:' + sha1(str.encode(page)).hexdigest()
        if qid not in query_context.keys():
            print(page + ' not in query context')
            continue
        clustering_results[page] = {'query_id': page, 'query_text': unquote(page.split('enwiki:')[1])}
        paralist = page_paras[page]
        paralist.sort()
        if query_type == 'title':
            query_vec = emb_model.encode([unquote(query_context[qid][0].split('enwiki:')[1])])[0]
        elif query_type == 'lead':
            query_vec = emb_model.encode([query_context[qid][1].strip()])[0]
        else:
            query_vec = np.mean([paravec_dict[p] for p in paralist], axis=0)
        X_page, parapairs = build_cluster_data(query_vec, paralist, paravec_dict)
        pair_scores = model(X_page)
        pair_scores = (pair_scores - torch.min(pair_scores)) / (torch.max(pair_scores) - torch.min(pair_scores))
        pair_score_dict = {}
        for i in range(len(parapairs)):
            pair_score_dict[parapairs[i]] = 1 - pair_scores[i].item()
        dist_mat = []
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
        clustering_results[page]['elements'] = paralist
        cl = AgglomerativeClustering(n_clusters=page_num_sections[page], affinity='precomputed', linkage='average')
        cl_labels = cl.fit_predict(dist_mat).tolist()
        clustering_results[page]['toplevel_cluster_idx'] = cl_labels
        cl_hq = AgglomerativeClustering(n_clusters=page_num_sections_hq[page], affinity='precomputed',
                                        linkage='average')
        cl_labels_hq = cl_hq.fit_predict(dist_mat).tolist()
        clustering_results[page]['hier_cluster_idx'] = cl_labels_hq
        print(page)

    return clustering_results


def get_baseline_clusters_from_run(paratext_file, query_context_file, cand_run_qrels_format, top_qrels, hier_qrels,
                                   emb_model_name='bert-base-uncased'):
    emb_model = SentenceTransformer(emb_model_name)
    paratexts = []
    paraids = []
    with open(paratext_file, 'r', encoding='utf8') as f:
        for l in f:
            paraids.append(l.split('\t')[0])
            paratexts.append(l.split('\t')[1].strip())
    paravecs = emb_model.encode(paratexts)
    paravec_dict = {}
    for i, para in enumerate(paraids):
        paravec_dict[para] = paravecs[i]
    with open(query_context_file, 'r') as f:
        query_context = json.load(f)
    page_paras = read_art_qrels(cand_run_qrels_format)
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
            if p in para_labels.keys():
                sec.add(para_labels[p])
            else:
                sec.add('nonrel')
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
            if p in para_labels_hq.keys():
                sec.add(para_labels_hq[p])
            else:
                sec.add('nonrel')
        page_num_sections_hq[page] = len(sec)
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    clustering_results_base, clustering_results_euc = {}, {}
    for page in page_paras.keys():
        qid = 'Query:' + sha1(str.encode(page)).hexdigest()
        if qid not in query_context.keys():
            print(page + ' not in query context')
            continue
        paralist = page_paras[page]
        paralist.sort()
        clustering_results_base[page] = {'query_id': page, 'query_text': unquote(page.split('enwiki:')[1])}
        clustering_results_euc[page] = {'query_id': page, 'query_text': unquote(page.split('enwiki:')[1])}
        paralist = page_paras[page]
        paralist.sort()
        dummy_query_vec = np.random.randn(768)
        X_page, parapairs = build_cluster_data(dummy_query_vec, paralist, paravec_dict)
        pair_baseline_scores = cos(X_page[:, 768:768 * 2], X_page[:, 768 * 2:])
        pair_euclid_scores = torch.sqrt(torch.sum((X_page[:, 768:768 * 2] - X_page[:, 768 * 2:]) ** 2, 1)).numpy()
        pair_baseline_scores = (pair_baseline_scores - torch.min(pair_baseline_scores)) / (
                torch.max(pair_baseline_scores) - torch.min(pair_baseline_scores))
        pair_euclid_scores = (pair_euclid_scores - np.min(pair_euclid_scores)) / (
                np.max(pair_euclid_scores) - np.min(pair_euclid_scores))
        pair_baseline_score_dict = {}
        pair_euclid_score_dict = {}
        for i in range(len(parapairs)):
            pair_baseline_score_dict[parapairs[i]] = 1 - pair_baseline_scores[i]
            pair_euclid_score_dict[parapairs[i]] = pair_euclid_scores[i]
        dist_base_mat = []
        dist_euc_mat = []
        for i in range(len(paralist)):
            rbase = []
            reuc = []
            for j in range(len(paralist)):
                if i == j:
                    rbase.append(0.0)
                    reuc.append(0.0)
                elif i < j:
                    rbase.append(pair_baseline_score_dict[paralist[i] + '_' + paralist[j]])
                    reuc.append(pair_euclid_score_dict[paralist[i] + '_' + paralist[j]])
                else:
                    rbase.append(pair_baseline_score_dict[paralist[j] + '_' + paralist[i]])
                    reuc.append(pair_euclid_score_dict[paralist[j] + '_' + paralist[i]])
            dist_base_mat.append(rbase)
            dist_euc_mat.append(reuc)

        cl = AgglomerativeClustering(n_clusters=page_num_sections[page], affinity='precomputed', linkage='average')
        clustering_results_base[page]['elements'] = paralist
        clustering_results_euc[page]['elements'] = paralist
        cl_base_labels = cl.fit_predict(dist_base_mat).tolist()
        cl_euclid_labels = cl.fit_predict(dist_euc_mat).tolist()
        clustering_results_base[page]['toplevel_clustering_idx'] = cl_base_labels
        clustering_results_euc[page]['toplevel_clustering_idx'] = cl_euclid_labels

        cl_hq = AgglomerativeClustering(n_clusters=page_num_sections_hq[page], affinity='precomputed',
                                        linkage='average')
        cl_base_labels_hq = cl_hq.fit_predict(dist_base_mat).tolist()
        cl_euclid_labels_hq = cl_hq.fit_predict(dist_euc_mat).tolist()
        clustering_results_base[page]['hier_clustering_idx'] = cl_base_labels_hq
        clustering_results_euc[page]['hier_clustering_idx'] = cl_euclid_labels_hq

        print(page)
    return clustering_results_base, clustering_results_euc