from gensim import corpora
from gensim.models import ldamodel
from nltk.corpus import stopwords
from collections import defaultdict
import argparse

def train_lda_tm(train_ptext_dict, num_topics, update, passes, token_dict_out, model_out_file):
    stops = stopwords.words('english')
    paraids = list(train_ptext_dict.keys())
    raw_docs = [train_ptext_dict[k] for k in paraids]
    pre_docs = [[word for word in doc.lower().split() if word not in stops] for doc in raw_docs]
    frequency = defaultdict(int)
    for d in pre_docs:
        for t in d:
            frequency[t] += 1
    texts = [[t for t in doc if frequency[t] > 1] for doc in pre_docs]
    token_dict = corpora.Dictionary(texts)
    corpus = [token_dict.doc2bow(text) for text in texts]

    model = ldamodel.LdaModel(corpus=corpus, id2word=token_dict, num_topics=num_topics, update_every=update, passes=passes)
    token_dict.save(token_dict_out)
    model.save(model_out_file)

def main():
    parser = argparse.ArgumentParser(description='Run CATS model')
    parser.add_argument('-tp', '--train_ptext')
    parser.add_argument('-nt', '--num_topics', type=int, default=100)
    parser.add_argument('-up', '--update', type=int)
    parser.add_argument('ps', '--passes', type=int)
    parser.add_argument('-td', '--token_dict_out')
    parser.add_argument('-mo', '--model_out')

    args = parser.parse_args()
    ptext_dict = {}
    with open(args.train_ptext, 'r') as f:
        for l in f:
            if len(l.split('\t')) > 1:
                ptext_dict[l.split('\t')[0]] = l.split('\t')[1].strip()
    train_lda_tm(ptext_dict, args.num_topics, args.update, args.passes, args.token_dict_out, args.model_out)

if __name__ == '__main__':
    main()