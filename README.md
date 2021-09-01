# Recommenders.jl
<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yng87.github.io/Recommenders.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://yng87.github.io/Recommenders.jl/dev) -->
[![Build Status](https://github.com/yng87/Recommenders.jl/workflows/CI/badge.svg)](https://github.com/yng87/Recommenders.jl/actions)

[WIP]
Collection of datasets and algorithms for recommendation from implicit feedback.

- Consistent interface for fit, predict and evaluate.
- Accepts Tables.jl - compatible data.

# Movielens 100k
 Random 80/20 split, all rating set to be 1.

| Model | Precision@10 | Recall@10 | nDCG@10 | Ref. |
|-------| -------------| ----------| ------- | ---- |
| Most popular | 0.1897 | 0.1182 | 0.2197 ||
| ItemkNN | 0.3284 | 0.2180 | 0.3986| [Item-based top-<i>N</i> recommendation algorithms](https://doi.org/10.1145/963770.963776) |
| Matrix factorization | 0.3657 | 0.2416 | 0.4402 | [Neural Collaborative Filtering vs. Matrix Factorization Revisited](http://arxiv.org/abs/2005.09683) |
| BPR | 0.3463 | 0.2316 | 0.4083 | [BPR: Bayesian Personalized Ranking from Implicit Feedback](http://arxiv.org/abs/1205.2618)


# Movielens 1M
Random 80/20 split, all rating set to be 1.

| Model | Precision@10 | Recall@10 | nDCG@10 | Ref. |
|-------| -------------| ----------| ------- | ---- |
| Most popular | 0.1816 | 0.06661 | 0.2022 ||
| ItemkNN | 0.3311 | 0.1489 | 0.3734 | [Item-based top-<i>N</i> recommendation algorithms](https://doi.org/10.1145/963770.963776) |
| Matrix factorization | | | | [Neural Collaborative Filtering vs. Matrix Factorization Revisited](http://arxiv.org/abs/2005.09683) |
| BPR | 0.3358 | 0.1432 | 0.3715 | [BPR: Bayesian Personalized Ranking from Implicit Feedback](http://arxiv.org/abs/1205.2618)
