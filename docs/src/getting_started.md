# Getting started

As an exmaple, we show how to train a Matrix factorization model on classic Movielens 100k dataset.
We do not heavily rely on utility functions in `Recommemders.jl`, and instead make use of Julia data ecosystem.

One needs to download the dataset from [official page](https://grouplens.org/datasets/movielens/100k/), and extract  data to any location you like.
## Data preparation
`Recommenders.jl` assumes the input data as tabular data.
To handle them, we rely on `Tables.jl` - abstract interface to handle all kinds of tabular objects.

`CSV` data, created by `CSV.jl`, is one such table object, and here we load movie rating data by using `CSV.File`:

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

For clarity, let's replace movie ids by its title
```julia
using Tables, TableOperations

movie = CSV.File(
    joinpath(<path/to/movielens100k>, "u.item"),
    delim = "|",
    header = [
        :movieid,
        :movie_title,
        :release_date,
        :video_release_date,
        :IMDbURL,
        :unknown,
        :Action,
        :Adventure,
        :Animation,
        :Childrens,
        :Comedy,
        :Crime,
        :Documentary,
        :Drama,
        :Fantasy,
        :FilmNoir,
        :Horror,
        :Musical,
        :Mystery,
        :Romance,
        :SciFi,
        :Thriller,
        :War,
        :Western,
    ],
)

id2title = Dict()
for row in Tables.rows(movie)
    id2title[row[:movieid]] = row[:movie_title]
end

rating = rating |> TableOperations.transform(movieid=x->id2title[x])
```

Since `Recommenders.jl` focus is implicit feedback dataset, we replace all the rating by unity.
```julia
rating = rating |> TableOperations.transform(rating=x->1)
```

Finally, split the dataset to train and test. Several data split methods are implemented in `Recommenders`, and the below is simple 80/20 split.
```julia
using Random
using Recommenders: ratio_split
Random.seed!(1234) # for reproducibility
train_table, test_table = ratio_split(rating, 0.8)
```

Check the first row entry
```julia
for row in Tables.rows(rating)
    print(row)
    break
end
```

```julia
Tables.ColumnsRow{TableOperations.Transforms{true, TableOperations.Transforms{true, CSV.File, NamedTuple{(:movieid,), Tuple{var"#1#2"}}}, NamedTuple{(:rating,), Tuple{var"#3#4"}}}}:
 :userid           196
 :movieid             "Kolya (1996)"
 :rating             1
 :timestamp  881250949
```

## Fit
Let's train the recommender model. Here we take matrix factorization model, but the fit API is similar for other models.
```julia
using Recommenders: ImplicitMF, fit!

dim = 128
use_bias = true
reg_coeff = 0.01

model = ImplicitMF(dim, use_bias, reg_coeff)

fit!(
    model,
    train_table,
    col_user = :userid, # specify user column
    col_item = :movieid, # specify item column
    n_epochs = 3,
    learning_rate = 0.01,
    n_negatives = 2, # number of negatives per positive sample
    verbose=1,
)
```
By setting `verbose=1`, one can see the training information.
```
[ Info: epoch=1: train_loss=Inf
[ Info: epoch=2: train_loss=0.6787517643400646
[ Info: epoch=3: train_loss=0.5865751731700346
```
## Predict
Let's get prediction for single user.
```julia
using Recommenders: predict_u2i

userid = 10
n = 3 # number of retrieved items
pred = predict_u2i(
    model,
    userid,
    n,
    drop_history = true, # whether to drop already consumed items
)
```

```julia
3-element Vector{String}:
 "Tomorrow Never Dies (1997)"
 "Return of the Jedi (1983)"
 "Star Trek: First Contact (1996)"
```

## Evaluate
We show how to evaluate the trained model.
Let's first make test set aggregated by users
```julia
user_actioned_items = Dict()
for row in Tables.rows(test_table)
    uid = row[:userid]
    iid = row[:movieid]
    if uid in keys(user_actioned_items)
        push!(user_actioned_items[uid], iid)
    else
        user_actioned_items[uid] = [iid]
    end
end
test_users = collect(keys(user_actioned_items))
ground_truth = collect(values(user_actioned_items))
```

Get predictions for all the user in test set
```julia
n = 3
preds = predict_u2i(
    model,
    test_users,
    n,
    drop_history = true,
)
```
Evaluation metrics are implemented as callable struct.
For instance, one can evaluate nDCG@10 averaged over all users by
```julia
using Recommenders: MeanNDCG
ndcg10 = MeanNDCG(10)
ndcg10(preds, ground_truth)
```
```
0.11942894185604025
```

Note that, in `Recommenders.jl`, this whole fit → predict → evaluate process is performed by the following `evaluate_u2i` API
```
model = ImplicitMF(dim, use_bias, reg_coeff)
metrics = [ndcg10]
n = 10

evaluate_u2i(
    model,
    train_table,
    test_table,
    metrics,
    n,
    col_user = :userid,
    col_item = :movieid,
    n_epochs = 3,
    learning_rate = 0.01,
    n_negatives = 2,
    verbose=1,
    drop_history = true,
)
```