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
    MeanNDCG,
    make_idmap!

ml100k = Movielens100k()
download(ml100k)
rating, _, _ = load_dataset(ml100k)

Random.seed!(1234);
train_table, test_table = ratio_split(rating, 0.8)

df_train = DataFrame(train_table)

userids = unique(df_train[!, :userid])
df_user = DataFrame(userid = userids, uidx = 1:length(userids))

itemids = unique(df_train[!, :movieid])
df_item = DataFrame(itemid = itemids, iidx = 1:length(itemids))

model = ItemkNN(10, 0.001, :bm25, true, true, true)

make_idmap!(model, df_user, df_item)
df_train[!, :uidx] = map(x -> model.user2uidx[x], df_train[!, :userid])
df_train[!, :iidx] = map(x -> model.item2iidx[x], df_train[!, :movieid])

prec10 = MeanPrecision(10)
recall10 = MeanRecall(10)
ndcg10 = MeanNDCG(10)
metrics = [prec10, recall10, ndcg10]

result = evaluate_u2i(
    model,
    df_train,
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
@test model.similarity !== nothing