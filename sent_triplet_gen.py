import numpy as np
from numpy.random import seed
seed(42)
import random
random.seed(42)
import csv
import os
import argparse

def read_art_qrels(art_qrels):
    art_qrels_dict = {}
    with open(art_qrels, 'r') as art:
        qrels_lines = art.readlines()
    for l in qrels_lines:
        title = l.split(' ')[0]
        if '/' in title:
            continue
        art_qrels_dict[title] = set()
    for l in qrels_lines:
        para = l.split(' ')[2]
        art_qrels_dict[title].add(para)
    print('art qrels read')
    titles = list(art_qrels_dict.keys())
    random.shuffle(titles)
    part1_titles = titles[:len(titles)//2]
    print('part 1 title obtained')
    part2_titles = titles[len(titles)//2:]
    print('part 2 title obtained')
    return art_qrels_dict, part1_titles, part2_titles

def sent_triple_gen(art_qrels, top_qrels, train_paratext, outdir):
    art_qrels_dict, part1_titles, part2_titles = read_art_qrels(art_qrels)
    print('There are '+str(len(part1_titles))+' articles to parse')
    paratext_dict = {}
    with open(train_paratext, 'r') as tp:
        for l in tp:
            l = l.strip()
            if len(l.split('\t')) < 2:
                paratext_dict[l.split('\t')[0]] = ''
            else:
                paratext_dict[l.split('\t')[0]] = l.split('\t')[1]
    print('Paratext read')
    top_qrels_dict = {}
    with open(top_qrels, 'r') as tq:
        tq_lines = tq.readlines()
    print('top qrels lines read')
    for l in tq_lines:
        q = l.split(' ')[0]
        title = q.split('/')[0]
        top_qrels_dict[title] = {}
    print('top qrels phase 1')
    for l in tq_lines:
        q = l.split(' ')[0]
        p = l.split(' ')[2]
        if q not in top_qrels_dict[title].keys():
            top_qrels_dict[title][q] = set([p])
        else:
            top_qrels_dict[title][q].add(p)
    print('Top qrels read')
    triples = []
    i = 0
    for title in part1_titles:
        paras = art_qrels_dict[title]
        if len(paras) < 3:
            continue
        para_label_dict = top_qrels_dict[title]
        if len(para_label_dict.keys()) < 2:
            continue
        for label in para_label_dict.keys():
            pos_paras = para_label_dict[label]
            if len(pos_paras) < 2:
                continue
            neg_paras = paras - pos_paras
            print('neg paras')
            anchor, pos = random.sample(pos_paras, 2)
            print('pos sample')
            neg = random.sample(neg_paras, 1)[0]
            print('neg sample')
            triples.append((title, paratext_dict[anchor].strip(), paratext_dict[pos].strip(),
                            paratext_dict[neg].strip(), anchor+'_'+pos+'_'+neg))
        i += 1
        if i%100==0:
            print(str(i)+' articles parsed')
    triples_count = len(triples)
    train_count = (triples_count * 8) // 10
    validation_count = triples_count // 10
    random.shuffle(triples)
    train_triples = triples[:train_count]
    validation_triples = triples[train_count:train_count+validation_count]
    test_triples = triples[train_count+validation_count:]
    print('Saving triples')
    with open(os.path.join(outdir, 'train.csv'), 'w', encoding='utf-8') as tr:
        writer = csv.writer(tr, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Article Title", "Sentence1", "Sentence2", "Sentence3", "Article Link"])
        for d in train_triples:
            writer.writerow(d)
    with open(os.path.join(outdir, 'validation.csv'), 'w', encoding='utf-8') as val:
        writer = csv.writer(val, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Article Title", "Sentence1", "Sentence2", "Sentence3", "Article Link"])
        for d in validation_triples:
            writer.writerow(d)
    with open(os.path.join(outdir, 'test.csv'), 'w', encoding='utf-8') as ts:
        writer = csv.writer(ts, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["Article Title", "Sentence1", "Sentence2", "Sentence3", "Article Link"])
        for d in test_triples:
            writer.writerow(d)

def main():
    parser = argparse.ArgumentParser(description='Generate passage triples for sentence-transformers')
    parser.add_argument('-aq', '--art_qrels', help='Path to article qrels')
    parser.add_argument('-tq', '--top_qrels', help='Path to top qrels')
    parser.add_argument('-tp', '--paratext', help='Path to paratext tsv')
    parser.add_argument('-od', '--output_dir', help='Path to output directory')
    args = parser.parse_args()
    sent_triple_gen(args.art_qrels, args.top_qrels, args.paratext, args.output_dir)

if __name__ == '__main__':
    main()