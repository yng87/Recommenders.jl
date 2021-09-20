# Getting started

We show how to train a Matrix factorization model on classic Movielens 100k dataset.
One needs to download the dataset from [official page](https://grouplens.org/datasets/movielens/100k/), and extract  data to any location you like.
## Data preparation
`Recommenders.jl` assumes the input data as tabular data.
To handle them, we rely on `Tables.jl` - abstract interface to handle all kinds of tabular objects.

`CSV` data is one such table object, and here we load movie rating data by using `CSV.File`:

```julia
using CSV
rating = CSV.File(
        joinpath(<path/to/movielens100k>, "u.data"),
        delim = "\t",
        header = [:userid, :movieid, :rating, :timestamp],
    )
```
The CSV table looks like
```julia
100000-element CSV.File{false}:
 CSV.Row: (userid = 196, movieid = 242, rating = 3, timestamp = 881250949)
 CSV.Row: (userid = 186, movieid = 302, rating = 3, timestamp = 891717742)
 CSV.Row: (userid = 22, movieid = 377, rating = 1, timestamp = 878887116)
 ⋮
```
One can instead use wrapper functions provided by `Recommenders` to do the same operations
```julia
using Recommenders: Movielens100k, load_dataset
ml100k = Movielens100k()
download(ml100k)
rating, _, _ = load_dataset(ml100k)
```

Since `Recommenders.jl` focus is implicit feedback dataset, we replace the all rating by unity.
Transformations on tabular object is done by `TableOperations` library as
```julia
using TableOperations
rating = rating |> TableOperations.transform(Dict(:rating=>x->1.))
```

Let's splint the dataset to train and test. Several data split methods are implemented in `Recommenders`, and the below is simple 80/20 split
```julia
using Recommenders: ratio_split
Random.seed!(1234) # for reproducibility
train_table, test_table = ratio_split(rating, 0.8)
```

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