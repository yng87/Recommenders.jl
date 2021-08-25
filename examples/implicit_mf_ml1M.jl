using DataFrames, TableOperations, Tables, Random
using TreeParzen
using Recommender:
    Movielens1M,
    load_dataset,
    ratio_split,
    ImplicitMF,
    evaluate_u2i,
    PrecisionAtK,
    RecallAtK,
    NDCG

ml1M = Movielens1M()
download(ml1M)
rating, user, movie = load_dataset(ml1M);

rating = rating |> TableOperations.filter(x -> Tables.getcolumn(x, :rating) >= 4)

Random.seed!(1234);
train_valid_table, test_table = ratio_split(rating, 0.8)

train_table, valid_table = ratio_split(train_valid_table, 0.8)
length(Tables.rows(train_table)),
length(Tables.rows(valid_table)),
length(Tables.rows(test_table))

prec10 = PrecisionAtK(10)
recall10 = RecallAtK(10)
ndcg10 = NDCG(10)
metrics = [prec10, recall10, ndcg10]

space = Dict(
    # :n_epochs => HP.Choice(:n_epochs, [1, 2]),
    :n_epochs => HP.Choice(:n_epochs, [32, 64, 128, 256]),
    :n_negatives => HP.QuantUniform(:n_negatives, 1.0, 16.0, 1.0),
    :learning_rate => HP.LogUniform(:learning_rate, log(1e-3), log(1.0)),
    :log2_dimension => HP.QuantUniform(:log2_dimension, 4.0, 9.0, 1.0),
    :use_bias => HP.Choice(:use_bias, [true, false]),
    :reg_coeff => HP.LogUniform(:reg_coeff, log(1e-3), log(1.0)),
)

function invert_output(params)
    @info params
    n_epochs = convert(Int, params[:n_epochs])
    n_negatives = convert(Int, params[:n_negatives])
    learning_rate = params[:learning_rate]
    dimension = convert(Int, 2^params[:log2_dimension])
    use_bias = params[:use_bias]
    reg_coeff = params[:reg_coeff]

    model = ImplicitMF(dimension, use_bias, reg_coeff)

    result = evaluate_u2i(
        model,
        train_table,
        valid_table,
        metrics,
        10,
        col_item = :movieid,
        n_epochs = n_epochs,
        n_negatives = n_negatives,
        learning_rate = learning_rate,
        drop_history = true,
        early_stopping_rounds = -1,
    )
    @info result
    return -result[end]
end

@info "Tuning start."
best = fmin(invert_output, space, 20, logging_interval = -1)
@info best

@info "Evaluate best model."

best_model =
    ImplicitMF(convert(Int, 2^best[:log2_dimension]), best[:use_bias], best[:reg_coeff])
result = evaluate_u2i(
    best_model,
    train_valid_table,
    test_table,
    metrics,
    10,
    col_item = :movieid,
    n_epochs = convert(Int, best[:n_epochs]),
    n_negatives = convert(Int, best[:n_negatives]),
    learning_rate = best[:learning_rate],
    drop_history = true,
    early_stopping_rounds = -1,
)
@info result
