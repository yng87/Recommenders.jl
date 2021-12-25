using Test
using DataFrames, TableOperations, Tables, Random
using Recommenders:
    Movielens100k,
    load_dataset,
    ratio_split,
    MostPopular,
    evaluate_u2i,
    predict_i2i,
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

model = MostPopular()
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

@test !(model.df_popular === nothing)

@test result[:ndcg10] >= 0
@test result[:precision10] >= 0
@test result[:recall10] >= 0

pred_u2i = predict_u2i(model, 1, 10)
@test length(pred_u2i) == 10

pred_u2i = predict_u2i(model, [1, 10], 10)
for p in pred_u2i
    @test length(p) == 10
end

pred_i2i = predict_i2i(model, 1, 10)
@test length(pred_i2i) == 10

pred_i2i = predict_i2i(model, [1, 10], 10)
for p in pred_i2i
    @test length(p) == 10
end

tempfile = tempname()
save_model(model, tempfile)
@test isfile(tempfile)
model = load_model(tempfile)
@test model.df_popular !== nothing
