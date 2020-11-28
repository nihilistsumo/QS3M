# CATS
Context Aware Triamese Similarity metric 

We propose a query Context-Aware Triamese Similarity (CATS) metric for query-specific clustering of paragraph-length texts. Our approach uses embeddings from BERT’s transformer to train a non-linear projection into a query-specific similarity space. We build on the idea of Siamese networks to include a third component, the query representation. Unlike other task-specific metric learning approaches, CATS accepts new unseen queries during test time. The empirical evaluation employs two TREC datasets to derive flat and hierarchical benchmarks for clustering. CATS achieves significant performance improvements over a recently published BERT-based reference method, as well as many baselines such as TF-IDF and topic models. This improvement translates to a 12% relative performance gain when extracting query-relevant subtopic clusters.

## How to run

