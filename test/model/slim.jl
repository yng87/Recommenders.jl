using Test
using DataFrames, TableOperations, Tables, Random
using Recommenders:
    Movielens100k,
    load_dataset,
    ratio_split,
    SLIM,
    evaluate_u2i,
    MeanPrecision,
    MeanRecall,
    MeanNDCG,
    EvaluateValidData

ml100k = Movielens100k()
download(ml100k)
rating, _, _ = load_dataset(ml100k)
rating = rating |> TableOperations.transform(Dict(:rating=>x->1.))

Random.seed!(1234);
train_table, test_table = ratio_split(rating, 0.8)



prec10 = MeanPrecision(10)
recall10 = MeanRecall(10)
ndcg10 = MeanNDCG(10)
metrics = [prec10, recall10, ndcg10]


model = SLIM(1e-4, 1e-4, 10)

result = evaluate_u2i(
    model,
    train_table,
    test_table,
    metrics,
    10,
    drop_history = true,
    col_item=:movieid,
    verbose = -1,
)

@test !(model.W === nothing)

@test result[:ndcg10] >= 0
@test result[:precision10] >= 0
@test result[:recall10] >= 0