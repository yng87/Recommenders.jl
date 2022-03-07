using DataFrames
using CSV
using Dates
using TableOperations
using Recommenders
using Recommenders: TopKMetric, MeanMetric, make_u2i_dataset
using Hyperopt
using Random
using Profile

struct AveragePrecision <: TopKMetric
    k::Int
    name::AbstractString

    AveragePrecision(k::Int) = new(k, "average_precision")
end

function (ap::AveragePrecision)(recommend_list, ground_truth)
    if recommend_list === nothing || length(recommend_list) == 0
        return 0.0
    end

    if ground_truth === nothing || length(ground_truth) == 0
        return 0.0
    end

    k = min(ap.k, length(recommend_list))
    truncated_recom = recommend_list[1:k]

    score = 0.0
    num_hits = 0.0
    for (i, p) in enumerate(truncated_recom)
        if p in ground_truth
            if i == 1 || !(p in truncated_recom[1:(i-1)])
                num_hits += 1
                score += num_hits / i
            end
        end
    end

    return score / min(length(ground_truth), k)
end

MAP(k) = MeanMetric(AveragePrecision(k))

function main()
    df_train = CSV.read(
        "../../kaggle_H_and_M_2022/dataset/transactions_train.csv",
        DataFrame,
        types = Dict("article_id" => String),
    )

    train_from_date = Date("2020-08-08")
    train_to_date = Date("2020-09-08")
    val_from_date = Date("2020-09-09")
    val_to_date = Date("2020-09-15")

    # subでは一週間ずつずらす
    # train_from_date = Date("2020-08-15")
    # train_to_date = Date("2020-09-15")
    # val_from_date = Date("2020-09-16")
    # val_to_date = Date("2020-09-22")

    df_val = filter(row -> val_from_date <= row.t_dat <= val_to_date, df_train)
    df_train = filter(row -> train_from_date <= row.t_dat <= train_to_date, df_train)

    df_train = unique(df_train[!, [:customer_id, :article_id]])
    df_train[!, :rating] .= 1.0

    map12 = MAP(12)
    map100 = MAP(100)
    recall100 = MeanRecall(100)
    metrics = [map12, map100, recall100]

    bohb = @hyperopt for i in 100,
        sampler in Hyperband(
            R = 50,
            η = 3,
            inner = BOHB(dims = [Hyperopt.Continuous(), Hyperopt.Continuous()]),
        ),
        k in LinRange(100, 500, 401),
        h in exp.(LinRange(log(1e-3), log(1e3), 100)),
        weighting in [:dummy, :tfidf, :bm25],
        normalize in [true, false],
        normalize_sim in [true, false]

        if state !== nothing
            k, h, weighting, normalize, normalize_sim = state
        end
        k = convert(Int, round(k))
        @show k, h, weighting, normalize, normalize_sim
        model = ItemkNN(k, h, weighting, normalize, normalize_sim, true)
        result = evaluate_u2i(
            model,
            df_train,
            df_val,
            metrics,
            100,
            col_user = :customer_id,
            col_item = :article_id,
            col_rating = :rating,
            drop_history = false,
        )
        @show result
        -result[:recall100], (k, h, weighting, normalize, normalize_sim)
    end
end

function profile_knn()
    df_train = CSV.read(
        "../../kaggle_H_and_M_2022/dataset/transactions_train.csv",
        DataFrame,
        types = Dict("article_id" => String),
    )

    train_from_date = Date("2020-08-08")
    train_to_date = Date("2020-09-08")
    val_from_date = Date("2020-09-09")
    val_to_date = Date("2020-09-15")

    # subでは一週間ずつずらす
    # train_from_date = Date("2020-08-15")
    # train_to_date = Date("2020-09-15")
    # val_from_date = Date("2020-09-16")
    # val_to_date = Date("2020-09-22")

    df_val = filter(row -> val_from_date <= row.t_dat <= val_to_date, df_train)
    df_train = filter(row -> train_from_date <= row.t_dat <= train_to_date, df_train)

    df_train = unique(df_train[!, [:customer_id, :article_id]])
    df_train[!, :rating] .= 1.0

    # TODO: for profileing, remove after done
    df_val = df_val[randperm(nrow(df_val))[1:1000], :]

    map12 = MAP(12)
    map100 = MAP(100)
    recall100 = MeanRecall(100)
    metrics = [map12, map100, recall100]

    (k, h, weighting, normalize, normalize_sim) =
        (1000, 17.47528400007683, :tfidf, false, false)

    k = convert(Int, round(k))
    @show k, h, weighting, normalize, normalize_sim
    model = ItemkNN(k, h, weighting, normalize, normalize_sim, true)
    fit!(
        model,
        df_train,
        col_user = :customer_id,
        col_item = :article_id,
        col_rating = :rating,
    )
    @info "Fnish fit."
    userids, _ = make_u2i_dataset(df_val, col_user = :customer_id, col_item = :article_id)
    @profile recoms = predict_u2i(model, userids, 100; drop_history = false)
    open("/tmp/prof.txt", "w") do s
        Profile.print(
            IOContext(s, :displaysize => (24, 500)),
            format = :flat,
            sortedby = :count,
            mincount = 50,
        )
    end
end

# main()
profile_knn()
