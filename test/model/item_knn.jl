using Test
using DataFrames, TableOperations, Tables, Random
using Recommenders:
    Movielens100k,
    load_dataset,
    ratio_split,
    ItemkNN,
    evaluate_u2i,
    MeanPrecision,
    MeanRecall,
    MeanNDCG

ml100k = Movielens100k()
download(ml100k)
rating, _, _ = load_dataset(ml100k)

Random.seed!(1234);
train_table, test_table = ratio_split(rating, 0.8)

prec10 = MeanPrecision(10)
recall10 = MeanRecall(10)
ndcg10 = MeanNDCG(10)
metrics = [prec10, recall10, ndcg10]

model = ItemkNN(10, 0.001, :bm25, true, true, true)
result = evaluate_u2i(
    model,
    train_table,
    test_table,
    metrics,
    10,
    col_user = :userid,
    col_item = :movieid,
    col_rating = :rating,
    drop_history = true,
)

@test !(model.similarity === nothing)

@test result[:ndcg10] >= 0
@test result[:precision10] >= 0
@test result[:recall10] >= 0
