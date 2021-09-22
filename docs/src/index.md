```@meta
CurrentModule = Recommenders
```
# Recommenders

This package aims to provide light-weight recommendation models, mainly for implicit feedback data. We want to provide
- consistent interface for model training and inference
- flexibility for input data with `Tables.jl` package, which offers simple, but powerful abstract interface for tabular data
- robust baseline metrics for classic datasets. The comparison of advanced recommendation models to these baselines turns out to be challenge [1, 2].


[1]: M. F. Dacrema et. al., [Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches](10.1145/3298689.3347058)

[2]: S. Rendle, [Evaluation Metrics for Item Recommendation under Sampling](http://arxiv.org/abs/1912.02263)

See [Getting started](@ref) for the usage.
