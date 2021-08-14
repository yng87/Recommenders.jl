mutable struct ItemkNN
    k::Int64
    shrink::Float64
    weighting::Union{Nothing,Symbol}
    weighting_at_inference::Bool
    normalize::Bool

    similarity::Any
    user_histories::Any
    user2uidx::Union{Dict,Nothing}
    item2iidx::Union{Dict,Nothing}
    iidx2item::Union{Dict,Nothing}

    ItemkNN(
        k::Int64,
        shrink::Float64,
        weighting::Union{Nothing,Symbol},
        weighting_at_inference::Bool,
        normalize::Bool,
    ) = new(
        k,
        shrink,
        weighting,
        weighting_at_inference,
        normalize,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

function fit!(
    model::ItemkNN,
    df::DataFrame,
    col_user::Symbol,
    col_item::Symbol,
    col_rating::Symbol,
)
    df_train, model.user2uidx = reindex_id_column!(df, col_user, suffix = "_ind")
    df_train, model.item2iidx = reindex_id_column!(df_train, col_item, suffix = "_ind")
    model.iidx2item = Dict(iidx => itemid for (itemid, iidx) in model.item2iidx)
    X = rows2sparse(
        df_train,
        col_user = "$(col_user)_ind",
        col_item = "$(col_item)_ind",
        col_rating = col_rating,
    )

    if !model.weighting_at_inference
        model.user_histories = X
    end

    if model.weighting == :tfidf
        X = tfidf(X)
    elseif model.weighting == :bm25
        X = bm25(X)
    end

    if model.weighting_at_inference
        model.user_histories = X
    end

    model.similarity = compute_similarity(X, model.k, model.shrink, model.normalize)
    return model
end

function predict(model::ItemkNN, userids, n::Int64; drop_history::Bool = false)
    uidxs = [get(model.user2uidx, userid, nothing) for userid in userids]
    preds = []
    for uidx in uidxs
        if uidx === nothing
            push!(preds, [])
        else
            push!(
                preds,
                predict_u2i(
                    model.similarity,
                    model.user_histories[uidx, :],
                    n,
                    drop_history = drop_history,
                ),
            )
        end
    end
    preds = [[model.iidx2item[iidx] for iidx in pred] for pred in preds]
    return preds
end

function evaluate(
    model::ItemkNN,
    df_train::DataFrame,
    df_test::DataFrame,
    metric,
    n::Int64;
    col_user::Symbol = :userid,
    col_item::Symbol = :itemid,
    col_rating::Symbol = :rating,
    drop_history::Bool = false,
)
    fit!(model, df_train, col_user, col_item, col_rating)
    xs, ys = make_u2i_dataset(df_test, col_user = col_user, col_item = col_item)
    recoms = predict(model, xs, n, drop_history = drop_history)
    result = metric(recoms, ys)
    return result
end
