# CATS: Context Aware Triamese Similarity metric 

We propose a query Context-Aware Triamese Similarity (CATS) metric for query-specific clustering of paragraph-length texts. Our approach uses embeddings from BERT’s transformer to train a non-linear projection into a query-specific similarity space. We build on the idea of Siamese networks to include a third component, the query representation. Unlike other task-specific metric learning approaches, CATS accepts new unseen queries during test time. The empirical evaluation employs two TREC datasets to derive flat and hierarchical benchmarks for clustering. CATS achieves significant performance improvements over a recently published BERT-based reference method, as well as many baselines such as TF-IDF and topic models. This improvement translates to a 12% relative performance gain when extracting query-relevant subtopic clusters.

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

## Parameters models.py

- -dd: Path to the dataset, change it to the directory where you downloaded the dataset.
- -qtr: Name of the query attention training file.
