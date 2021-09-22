# Recommenders.jl
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yng87.github.io/Recommenders.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://yng87.github.io/Recommenders.jl/dev)
[![Build Status](https://github.com/yng87/Recommenders.jl/workflows/CI/badge.svg)](https://github.com/yng87/Recommenders.jl/actions)

This package aims to provide light-weight recommendation models, mainly for implicit feedback data. We want to provide
- consistent interface for model training and inference
- flexibility for input data with `Tables.jl` package, which offers simple, but powerful abstract interface for tabular data
- robust baseline metrics for classic datasets. The comparison of advanced recommendation models to these baselines turns out to be challenge.

See [Getting started](https://yng87.github.io/Recommenders.jl/stable/getting_started/) for the usage.

# Implemented algorithms

| Model | Ref. | Note |
|-------|------|------|
| Most popular | | |
| ItemkNN | [Item-based top-<i>N</i> recommendation algorithms](https://doi.org/10.1145/963770.963776) | |
| Matrix factorization | [Neural Collaborative Filtering vs. Matrix Factorization Revisited](http://arxiv.org/abs/2005.09683) | |
| BPR MF | [BPR: Bayesian Personalized Ranking from Implicit Feedback](http://arxiv.org/abs/1205.2618) | Simple SGD with negative samping is implemented instead of the original learnBPR algorithm.|
| SLIM ElasticNet | [Efficient Top-N Recommendation by Linear Regression](https://www.slideshare.net/MarkLevy/efficient-slides) <br /> [SLIM: Sparse Linear Methods for Top-N Recommender Systems](http://glaros.dtc.umn.edu/gkhome/node/774) | |
| Random walk | [Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time](http://dl.acm.org/citation.cfm?doid=3178876.3186183) | |

# Evaluation
## Movielens 100k
 Random 80/20 split, treated as implicit feedback.

| Model | Precision@10 | Recall@10 | nDCG@10 | Best parameters |
|-------| -------------| ----------| ------- | ---- |
| Most popular | 0.1897 | 0.1182 | 0.2197 | |
| ItemkNN | 0.3284 | 0.2180 | 0.3986| <details><summary></summary><p>weighting: BM25<br /> weighting_at_inference: false<br />topk: 358<br />normalize: true<br />normalize_similarity: true<br />shrink: 0.00258278</p></details> |
| Matrix factorization | 0.3657 | 0.2416 | 0.4402 |<details><summary></summary><p>dimension: 32<br />l2_coeff: 0.01797358830471941<br /> n_epochs: 128<br />n_negatives: 7<br />learning_rate: 0.017678089718746345</p></details>|
| BPR MF| 0.3463 | 0.2316 | 0.4083 |<details><summary></summary><p>dimension: 512<br />l2_coeff: 0.015587614364453028<br /> n_epochs: 128<br />n_negatives: 12<br />learning_rate: 0.007785000886303088</p></details>|
| SLIM ElasticNet | 0.3626 | 0.2385 | 0.4359 |<details><summary></summary><p>k: 908<br />λminratio: 0.029052468222707274<br /> l1_ratio: 0.003308685140740372</p></details>|
| Random walk | 0.2634 | 0.1828 | 0.3255 | <details><summary></summary><p>pixie_walk_length_scaling: false<br />pixie_multi_hit_boosting: false<br /> terminate_prob: 0.9<br />total_walk_length: 51307<br />min_high_visited_candidates: 500<br />high_visited_count_threshold: 64</p></details>


## Movielens 1M
Random 80/20 split, treated as implicit feedback.

| Model | Precision@10 | Recall@10 | nDCG@10 | Best parameters |
|-------| -------------| ----------| ------- | ---- |
| Most popular | 0.1816 | 0.06661 | 0.2022 | |
| ItemkNN | 0.3311 | 0.1489 | 0.3734 | <details><summary></summary><p>weighting: TF-IDF<br />topk: 43<br />normalize: true<br />normalize_similarity: true<br />shrink: 0.453735</p></details>  |
| Matrix factorization | 0.3752 | 0.1687 | 0.4203 |<details><summary></summary><p>dimension: 128<br />l2_coeff: 0.005720177108336541<br /> n_epochs: 256<br />n_negatives: 18<br />learning_rate: 0.0012750705664730715</p></details>|
| BPR MF| 0.3358 | 0.1432 | 0.3715 | <details><summary></summary><p>dimension: 128<br />l2_coeff: 0.017941536727080445<br /> n_epochs: 256<br />n_negatives: 6<br />learning_rate: 0.0014264826446678315</p></details> |
| SLIM ElasticNet | 0.3831 | 0.1792 | 0.4334 | <details><summary></summary><p>k: 444<br />λminratio: 3.968133599128073e-5<br /> l1_ratio: 2.500068207067635e-5</p></details>|
| Random walk | 0.2314 | 0.09729 | 0.2665 | <details><summary></summary><p>pixie_walk_length_scaling: false<br />pixie_multi_hit_boosting: false<br /> terminate_prob: 0.9<br />total_walk_length: 712047<br />min_high_visited_candidates: None<br />high_visited_count_threshold: Inf</p></details>