# Recommenders.jl
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://yng87.github.io/Recommenders.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://yng87.github.io/Recommenders.jl/dev)
[![Build Status](https://github.com/yng87/Recommenders.jl/workflows/CI/badge.svg)](https://github.com/yng87/Recommenders.jl/actions)

[WIP]
Collection of datasets and algorithms for recommendation from implicit feedback.

- Consistent interface for fit, predict and evaluate.
- Accepts Tables.jl - compatible data.

# Usage
## Data preparation
`Recommenders.jl` assumes the input data as `table` - each row corresponds to one observed event. For instamce, if user of id 1 consumes item of id 100, the table row reads (1, 10) (or (1, 10, 3.5) where 3.5 is explicit feedback). 

The input data structure is general `Tables.jl` - compatible data. For instance one can use CSV data
```julia
using CSV
table = CSV.File(<path/to/csv>)
```
We also provide dataset loader for some popular datasets. For instance, Movielens 100k dataset is used by
```julia
using TableOperations
using Recommenders: Movielens100k, load_dataset
ml100k = Movielens100k()
download(ml100k)
rating, user, movie = load_dataset(ml100k)
# make it implicit feedback data
rating = rating |> TableOperations.transform(Dict(:rating=>x->1.))
```
Several data split methods are implemented. Below is the simple 80/20 split
```julia
train_table, test_table = ratio_split(rating, 0.8)
```
See docs for other methods.

## Fit
The models are any struct of type `<: AbstractRecommender`. The model training is performed by
```julia
fit!(model::AbstractRecommender, table; kwargs...)
```
where `kwargs` is model-dependent keyword arguments. Let's see the example of matrix factorization model
```julia
dim = 128
use_bias = true
reg_coeff = 0.01

model = ImplicitMF(dim, use_bias, reg_coeff)

fit!(model::ImplicitMF,
    train_table;
    col_user = :userid,
    col_item = :itemid,
    n_epochs = 32,
    learning_rate = 0.1,
    n_negatives = 2,
)
```

## Predict
Currently only user-to-item prediction is available.
```julia
# For single user
userid = 1
n = 10 # number of retrieved items
pred = predict_u2i(
    model,
    userid,
    n
    drop_history = true, # whether to drop already consumed items
)

# For multiple users
userids = [1, 2, 3]
preds = predict_u2i(
    model,
    userids,
    n
    drop_history = true,
)
```

## Evaluate
Evaluation metrics are implemented as callable struct.
For instance, one can evaluate nDCG@10 averaged over all users by
```julia
using Recommenders: MeanPrecision, MeanNDCG
ndcg10 = MeanNDCG(10)

ground_truths = [[10, 20], [10, 30], [40]]

ndcg10(preds, ground_truths)
```

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