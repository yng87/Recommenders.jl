using Test
using DataFrames, TableOperations, Tables, Random
using Recommenders:
    Movielens100k,
    load_dataset,
    ratio_split,
    BPR,
    evaluate_u2i,
    MeanPrecision,
    MeanRecall,
    MeanNDCG,
    EvaluateValidData

ml100k = Movielens100k()
download(ml100k)
rating, _, _ = load_dataset(ml100k)

Random.seed!(1234);
train_table, test_table = ratio_split(rating, 0.8)

prec10 = MeanPrecision(10)
recall10 = MeanRecall(10)
ndcg10 = MeanNDCG(10)
metrics = [prec10, recall10, ndcg10]

cb = EvaluateValidData(ndcg10, test_table, 2)

model = BPR(16, 0.01)
result = evaluate_u2i(
    model,
    train_table,
    test_table,
    metrics,
    10,
    callbacks = [cb],
    col_item = :movieid,
    n_epochs = 2,
    n_negatives = 1,
    learning_rate = 0.01,
    drop_history = true,
    verbose = 1,
)

@test !(model.user_embedding === nothing)
@test !(model.item_embedding === nothing)

@test result[:ndcg10] >= 0
@test result[:precision10] >= 0
@test result[:recall10] >= 0

tempfile = tempname()
save_model(model, tempfile)
@test isfile(tempfile)
model = load_model(tempfile)
@test model.user_embedding !== nothing
@test model.item_embedding !== nothing