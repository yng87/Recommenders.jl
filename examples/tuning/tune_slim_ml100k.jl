using DataFrames, TableOperations, Tables, Random, Logging
using TreeParzen
using Recommenders:
    Movielens100k,
    load_dataset,
    ratio_split,
    SLIM,
    evaluate_u2i,
    MeanPrecision,
    MeanRecall,
    MeanNDCG

function main()
    ml100k = Movielens100k()
    download(ml100k)
    rating, _, _ = load_dataset(ml100k)

    rating = rating |> TableOperations.transform(Dict(:rating => x -> 1.0))
    for (i, row) in enumerate(Tables.rows(rating))
        println(row)
        if i >= 4
            break
        end
    end

    Random.seed!(1234)
    train_valid_table, test_table = ratio_split(rating, 0.8)
    train_table, valid_table = ratio_split(train_valid_table, 0.8)

    prec10 = MeanPrecision(10)
    recall10 = MeanRecall(10)
    ndcg10 = MeanNDCG(10)
    metrics = [prec10, recall10, ndcg10]

    space = Dict(
        :λminratio => HP.LogUniform(:λminratio, log(1e-6), log(0.5)),
        :l1_ratio => HP.LogUniform(:l1_ratio, log(1e-6), log(0.9999)),
        :k => HP.QuantUniform(:k, 1.0, 1000.0, 1.0),
    )

    function invert_output(params)
        @info params
        λminratio = params[:λminratio]
        l1_ratio = params[:l1_ratio]
        k = convert(Int, params[:k])

        model = SLIM(l1_ratio, λminratio, k)

        result = evaluate_u2i(
            model,
            train_table,
            valid_table,
            metrics,
            10,
            col_item = :movieid,
            drop_history = true,
        )
        @info result
        return -result[:ndcg10]
    end

    @info "Tuning start."
    best = fmin(invert_output, space, 100, logging_interval = -1)
    @info best

    @info "Evaluate best model."

    best_model = SLIM(best[:l1_ratio], best[:λminratio], convert(Int, best[:k]))
    result = evaluate_u2i(
        best_model,
        train_valid_table,
        test_table,
        metrics,
        10,
        col_item = :movieid,
        drop_history = true,
    )
    @info result

end

io = open("log_param_search_slim_ml00k.txt", "w+")

logger = SimpleLogger(io)
global_logger(logger)
main()

close(io)
