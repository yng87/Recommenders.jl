using DataFrames, TableOperations, Tables, Random, TreeParzen
using Recommenders:
    Movielens1M,
    load_dataset,
    ratio_split,
    Randomwalk,
    evaluate_u2i,
    MeanPrecision,
    MeanRecall,
    MeanNDCG,
    fit!,
    predict_u2i,
    make_u2i_dataset,
    name

function main()
    ml1M = Movielens1M()
    download(ml1M)
    rating, _, _ = load_dataset(ml1M)

    rating = rating |> TableOperations.transform(Dict(:rating => x -> 1.0))

    Random.seed!(1234)
    train_valid_table, test_table = ratio_split(rating, 0.8)
    train_table, valid_table = ratio_split(train_valid_table, 0.8)

    prec10 = MeanPrecision(10)
    recall10 = MeanRecall(10)
    ndcg10 = MeanNDCG(10)
    metrics = [prec10, recall10, ndcg10]

    space = Dict(
        :terminate_prob => HP.QuantUniform(:terminate_prob, 0.1, 0.9, 0.1),
        :total_walk_length => HP.LogUniform(:total_walk_length, log(1e3), log(1e6)),
        :min_high_visited_candidates =>
            HP.Choice(:min_high_visited_candidates, [10, 50, 100, 250, 500, 1000]),
        :high_visited_count_threshold =>
            HP.Choice(:high_visited_count_threshold, [2, 4, 8, 16, 32, 64, Inf]),
        :pixie_walk_length_scaling =>
            HP.Choice(:pixie_walk_length_scaling, [true, false]),
        :pixie_multi_hit_boosting =>
            HP.Choice(:pixie_multi_hit_boosting, [true, false]),
    )

    model = Randomwalk()
    fit!(model, train_table, col_item = :movieid)
    userids_valid, gts_valid = make_u2i_dataset(valid_table, col_item = :movieid)

    function invert_output(params)
        terminate_prob = params[:terminate_prob]
        total_walk_length = round(Int, params[:total_walk_length])
        min_high_visited_candidates = convert(Int, params[:min_high_visited_candidates])
        high_visited_count_threshold = params[:high_visited_count_threshold]
        if high_visited_count_threshold < Inf
            high_visited_count_threshold = convert(Int, high_visited_count_threshold)
        end
        pixie_walk_length_scaling = params[:pixie_walk_length_scaling]
        pixie_multi_hit_boosting = params[:pixie_multi_hit_boosting]

        recoms = predict_u2i(
            model,
            userids_valid,
            10,
            col_user = :userid,
            col_item = :movieid,
            col_rating = :rating,
            drop_history = true,
            terminate_prob = terminate_prob,
            total_walk_length = total_walk_length,
            min_high_visited_candidates = min_high_visited_candidates,
            high_visited_count_threshold = high_visited_count_threshold,
            pixie_walk_length_scaling = pixie_walk_length_scaling,
            pixie_multi_hit_boosting = pixie_multi_hit_boosting,
        )
        result = Dict()
        for metric in metrics
            result[Symbol(name(metric))] = metric(recoms, gts_valid)
        end
        result = NamedTuple(result)
        @info params, result
        return -result[:ndcg10]
    end

    @info "Tuning start."
    best = fmin(invert_output, space, 100, logging_interval = -1)
    @info best

    @info "Evaluate best model."

    terminate_prob = best[:terminate_prob]
    total_walk_length = round(Int, best[:total_walk_length])
    min_high_visited_candidates = convert(Int, best[:min_high_visited_candidates])
    high_visited_count_threshold = best[:high_visited_count_threshold]
    if high_visited_count_threshold < Inf
        high_visited_count_threshold = convert(Int, high_visited_count_threshold)
    end
    pixie_walk_length_scaling = best[:pixie_walk_length_scaling]
    pixie_multi_hit_boosting = best[:pixie_multi_hit_boosting]

    model = Randomwalk()
    result = evaluate_u2i(
        model,
        train_valid_table,
        test_table,
        metrics,
        10,
        col_user = :userid,
        col_item = :movieid,
        col_rating = :rating,
        drop_history = true,
        terminate_prob = terminate_prob,
        total_walk_length = total_walk_length,
        min_high_visited_candidates = min_high_visited_candidates,
        high_visited_count_threshold = high_visited_count_threshold,
        pixie_walk_length_scaling = pixie_walk_length_scaling,
        pixie_multi_hit_boosting = pixie_multi_hit_boosting,
    )
    @info result

end

main()
