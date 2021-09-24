# Evaluation metrics

The performance measure of recommendation compares the each recommended item list to the ground truth items, and averages them over whole recommendation list. Here is an example

```julia
# Two recommendation, each of which has a collection of predicted item ids with descending order of scores.
recommends = [
    [1, 2],
    [4, 5]
]
# Ground truth item ids corresponding to each recommendation
ground_truth = [
    [1],
    [4, 5]
]
```
The Precision@2 for the first entry is ``1/2=0.5``, while the second is ``2/2=1``. Therefore, the mean Precision@2 is ``0.75``.

In `Recommenders.jl`, this computation is done by
```julia
Recommenders: MeanPrecision
prec2 = MeanPrecision(2) # metrics are implemented as callable struct
prec2(recommends, ground_truth)
# 0.75
```

Currently the following metrics are implemented. They are all descendent of `MeanMetric` type.
```@autodocs
Modules = [Recommenders]
Order   = [:type, :function]
Pages   = ["metric.jl"]
```