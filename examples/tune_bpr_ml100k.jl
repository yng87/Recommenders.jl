using DataFrames, TableOperations, Tables, Random
using TreeParzen
using Recommenders:
    Movielens100k,
    load_dataset,
    ratio_split,
    BPR,
    evaluate_u2i,
    MeanPrecision,
    MeanRecall,
    MeanNDCG

function main()
    ml100k = Movielens100k()
    download(ml100k)
    rating, _, _ = load_dataset(ml100k)

    Random.seed!(1234)
    train_valid_table, test_table = ratio_split(rating, 0.8)
    train_table, valid_table = ratio_split(train_valid_table, 0.8)

    prec10 = MeanPrecision(10)
    recall10 = MeanRecall(10)
    ndcg10 = MeanNDCG(10)
    metrics = [prec10, recall10, ndcg10]

    space = Dict(
        # :n_epochs => HP.Choice(:n_epochs, [1, 2]),
        :learning_rate => HP.LogUniform(:learning_rate, log(1e-4), log(1e-1)),
        :dimension => HP.QuantUniform(:dimension, 16., 128., 2.),
        :reg_coeff => HP.LogUniform(:reg_coeff, log(1e-9), log(1e-1)),
    )

    function invert_output(params)
        @info params
        learning_rate = params[:learning_rate]
        dimension = convert(Int, params[:dimension])
        reg_coeff = params[:reg_coeff]

        model = BPR(dimension, reg_coeff)

        result = evaluate_u2i(
            model,
            train_table,
            valid_table,
            metrics,
            10,
            valid_table=valid_table,
            valid_metric=ndcg10,
            col_item = :movieid,
            n_epochs = 1000,
            steps_in_epoch = 60000,
            early_stopping_rounds = 50,
            learning_rate = learning_rate,
            drop_history = true,
            verbose=-1,
        )
        @info result
        return -result[:ndcg10]
    end

    @info "Tuning start."
    best = fmin(invert_output, space, 100, logging_interval = -1)
    @info best

    # @info "Evaluate best model."

    # best_model = BPR(convert(Int, best[:dimension]), best[:reg_coeff])
    # result = evaluate_u2i(
    #     best_model,
    #     train_valid_table,
    #     test_table,
    #     metrics,
    #     10,
    #     col_item = :movieid,
    #     tolerance = best[:tolerance],
    #     learning_rate = best[:learning_rate],
    #     drop_history = true,
    #     early_stopping_rounds = -1,
    # )
    # @info result

end

main()
