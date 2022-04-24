# QS3M: Query-Specific Siamese Similarity metric 

We propose a Query-Specific Siamese Similarity Metric (QS3M) for query-specific clustering of text documents. Our approach uses fine-tuned BERT embeddings to train a non-linear projection into a query-specific similarity space. We build on the idea of Siamese networks, but include a third component, a representation of the query. QS3M is able to model the fine-grained similarity between text passages about the same broad topic and also generalizes to new unseen queries during evaluation. The empirical evaluation for clustering employs two TREC datasets and a set of academic abstracts from ArXiv. When used to obtain query-relevant clusters, QS3M achieves a 12\% performance improvement on the TREC datasets over a strong BERT-based reference method, as well as many baselines such as TF-IDF and topic models. A similar improvement is observed for the arXiv dataset suggesting the general applicability of QS3M to different domains. Qualitative evaluation is carried out to gain insight into the strengths and limitations of the model.

## Quickstart

1. Download the dataset for training and evaluation: https://drive.google.com/drive/folders/1fs3e59jdqBgl-kkmK7yZERVOA_rtXqOm?usp=sharing

2. Clone this repository, move to the directory and add the current directory to the python classpath
```
git clone https://github.com/nihilistsumo/CATS.git
cd CATS
export PYTHONPATH=.
```

3. Train the CATS model
```
python3 model/models.py -dd path/to/downloaded/data/ --save
```
This trains the CATS model with the default parameters and saves the trained model in the "saved_models" directory inside the current directory.

4. Evaluate the trained model
```
python3 eval/eval_model.py -dd path/to/downloaded/data/ -mp saved_models/name-of-the-trained-model.model
```
This evaluates the trained model on two test datasets (TRECCAR benchmark Y1 train and test). Specify the model name saved in "saved_models" directory trained in the previous step.

To train and evaluate other variations of CATS, please make necessary changes to parameters. A detailed description of various parameters can be found in the following section.

## Important parameters models.py

- -dd: Path to the dataset, change it to the directory where you downloaded the dataset.
- -qtr/ qt: Name of the query attention training file. It follows the below format:

Query_ID Passage1_ID Passage2_ID Binary_label

For example:

QID	P1	P2	label

Query:fb8aada8fa759f397d76bccb456f03b8c6b04f5e	07b06638e380f0e265df192da81907cb90a14731	90c7c1f216d1de7eee9b10978defa34a86ef3aab	1

Query:4243236b74ede0e38b5e072e3c11063ced49c598	b344fff55993cb37f29eaa8900369bf840c81e2f	29646695b42c392109062388a79f2f873620e9e6	0

Query:c4539c1ce700a1c9097ffded910ae5a16433a2f7	433a61f8d76e73309ce3e224b955e72e89cf7825	6a6d6b4cc930bee716370cc5bc045411063ae103	0

....
- -trp/ tp: List of passage IDs in form of numpy array.
- -trv/ tv: List of passage embedding vectors in form of numpy array in the same order of passage ID list.
- -trq/ tq: List of query IDs in form of numpy array.
- -trqv/ tqv: List of query embedding vectors in form of numpy array in the same order of query ID list.
- -ct: Different variations of query-specific similarity algorithms. Options are cats, scaled (CAVS), qscale (experimental) and abl (used for ablation study).
- --cache: Use intermediate data saved in the cache folder. Useful for experimenting with hyperparameters.
- --save: Save the trained model.

## Important parameters eval_model.py
  
- -qt: Name of the query attention file with the same format as before. Used for clustering evaluation. 
- -pp: Name of parapairs json file used for pairwise evaluation. It follows the below format:
  {'Query_ID': {'parapairs': [passage1ID_passage2ID, passage1ID_passage3ID, ...], 'labels': [1, 0, ...]}, ...}
- -aql: Name of article level qrels in TRECCAR format.
- -tql: Name of toplevel level qrels in TRECCAR format.
- -hql: Name of toplevel level qrels in TRECCAR format.
- -mp: Path to the saved models that we want to evaluate.
- -mt: Type of the saved model, using the same options as -ct option of models.py.
