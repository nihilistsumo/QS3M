# CATS: Context Aware Triamese Similarity metric 

We propose a query Context-Aware Triamese Similarity (CATS) metric for query-specific clustering of paragraph-length texts. Our approach uses embeddings from BERT’s transformer to train a non-linear projection into a query-specific similarity space. We build on the idea of Siamese networks to include a third component, the query representation. Unlike other task-specific metric learning approaches, CATS accepts new unseen queries during test time. The empirical evaluation employs two TREC datasets to derive flat and hierarchical benchmarks for clustering. CATS achieves significant performance improvements over a recently published BERT-based reference method, as well as many baselines such as TF-IDF and topic models. This improvement translates to a 12% relative performance gain when extracting query-relevant subtopic clusters.

## Quickstart

1. Download the dataset for training and evaluation: https://drive.google.com/drive/folders/1fs3e59jdqBgl-kkmK7yZERVOA_rtXqOm?usp=sharing

2. Clone this repository and move to the directory
```
git clone https://github.com/nihilistsumo/CATS.git
cd CATS
```

3. Train the CATS model
```
python3 model/models.py -dd path/to/downloaded/data --save
```

4. Evaluate the trained model
```
python3 eval/eval_model.py -dd path/to/downloaded/data
