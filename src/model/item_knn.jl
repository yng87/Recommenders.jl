mutable struct ItemkNN <: AbstractRecommender
    k::Int64
    shrink::Float64
    weighting::Union{Nothing,Symbol}
    weighting_at_inference::Bool
    normalize::Bool
    normalize_similarity::Bool

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
        normalize_similarity::Bool,
    ) = new(
        k,
        shrink,
        weighting,
        weighting_at_inference,
        normalize,
        normalize_similarity,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

function fit!(model::ItemkNN, table; kwargs...)
    col_user = get(kwargs, :col_user, :userid)
    col_item = get(kwargs, :col_item, :itemid)
    col_rating = get(kwargs, :col_rating, :rating)

    table, model.user2uidx, model.item2iidx, model.iidx2item =
        make_idmap(table, col_user = col_user, col_item = col_item)

    X = rows2sparse(
        table,
        col_user = col_user,
        col_item = col_item,
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

    model.similarity = compute_similarity(
        X,
        model.k,
        model.shrink,
        model.normalize,
        model.normalize_similarity,
    )
    return model
end

function predict_u2i(
    model::ItemkNN,
    userid::Union{AbstractString,Int},
    n::Int64;
    drop_history = false,
    kwargs...,
)
    if userid in keys(model.user2uidx)
        uidx = model.user2uidx[userid]
        pred = predict_u2i(
            model.similarity,
            model.user_histories[uidx, :],
            n,
            drop_history = drop_history,
        )
        return [model.iidx2item[iidx] for iidx in pred]

    else
        return []
    end
end
