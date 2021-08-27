using DataFrames, TableOperations, Tables, Random
using TreeParzen
using Recommenders:
    Movielens1M,
    load_dataset,
    ratio_split,
    BPR,
    evaluate_u2i,
    MeanPrecision,
    MeanRecall,
    MeanNDCG

function main()
    ml1M = Movielens1M()
    download(ml1M)
    rating, _, _ = load_dataset(ml1M)

    Random.seed!(1234)
    train_valid_table, test_table = ratio_split(rating, 0.8)
    train_table, valid_table = ratio_split(train_valid_table, 0.8)

    prec10 = MeanPrecision(10)
    recall10 = MeanRecall(10)
    ndcg10 = MeanNDCG(10)
    metrics = [prec10, recall10, ndcg10]

    space = Dict(
        # :n_epochs => HP.Choice(:n_epochs, [1, 2]),
        :n_epochs => HP.Choice(:n_epochs, [32, 64, 128, 256]),
        :learning_rate => HP.LogUniform(:learning_rate, log(1e-3), log(1.0)),
        :log2_dimension => HP.QuantUniform(:log2_dimension, 4.0, 9.0, 1.0),
        :reg_coeff => HP.LogUniform(:reg_coeff, log(1e-3), log(1.0)),
    )

    function invert_output(params)
        @info params
        n_epochs = convert(Int, params[:n_epochs])
        learning_rate = params[:learning_rate]
        dimension = convert(Int, 2^params[:log2_dimension])
        reg_coeff = params[:reg_coeff]

        model = BPR(dimension, reg_coeff)

        result = evaluate_u2i(
            model,
            train_table,
            valid_table,
            metrics,
            10,
            col_item = :movieid,
            n_epochs = n_epochs,
            learning_rate = learning_rate,
            drop_history = true,
            early_stopping_rounds = -1,
        )
        @info result
        return -result[:ndcg10]
    end

    @info "Tuning start."
    best = fmin(invert_output, space, 20, logging_interval = -1)
    @info best

    @info "Evaluate best model."

    best_model = BPR(convert(Int, 2^best[:log2_dimension]), best[:reg_coeff])
    result = evaluate_u2i(
        best_model,
        train_valid_table,
        test_table,
        metrics,
        10,
        col_item = :movieid,
        n_epochs = convert(Int, best[:n_epochs]),
        learning_rate = best[:learning_rate],
        drop_history = true,
        early_stopping_rounds = -1,
    )
    @info result

end

main()
