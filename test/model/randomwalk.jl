using Test
using DataFrames, TableOperations, Tables, Random
using Recommenders:
    Movielens100k,
    load_dataset,
    ratio_split,
    Randomwalk,
    evaluate_u2i,
    MeanPrecision,
    MeanRecall,
    MeanNDCG,
    EvaluateValidData

ml100k = Movielens100k()
download(ml100k)
rating, _, _ = load_dataset(ml100k)
rating = rating |> TableOperations.transform(Dict(:rating => x -> 1.0))

Random.seed!(1234);
train_table, test_table = ratio_split(rating, 0.8)



prec10 = MeanPrecision(10)
recall10 = MeanRecall(10)
ndcg10 = MeanNDCG(10)
metrics = [prec10, recall10, ndcg10]


model = Randomwalk()

result = evaluate_u2i(
    model,
    train_table,
    test_table,
    metrics,
    10,
    drop_history = true,
    col_item = :movieid,
    terminate_prob = 0.5,
    total_walk_length = 100000,
    min_high_visited_candidates = 500,
    high_visited_count_threshold = 4,
    pixie_walk_length_scaling = true,
    pixie_multi_hit_boosting = true,
)

@test !(model.adjacency_list === nothing)
@test !(model.offsets === nothing)

@test result[:ndcg10] >= 0
@test result[:precision10] >= 0
@test result[:recall10] >= 0
