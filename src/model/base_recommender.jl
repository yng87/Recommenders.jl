abstract type AbstractRecommender end

function fit!(model::AbstractRecommender, table; kwargs...)
    throw("Not implemented.")
end

function predict_u2i(model::AbstractRecommender, userids, n::Int64; kwargs...)
    throw("Not implemented.")
end

function evaluate_u2i(
    model::AbstractRecommender,
    train_table,
    test_table,
    metric::AbstractMetric,
    n::Int64;
    kwargs...,
)
    col_user = get(kwargs, :col_user, :userid)
    col_item = get(kwargs, :col_item, :itemid)
    fit!(model, train_table; kwargs...)
    xs, ys = make_u2i_dataset(test_table, col_user = col_user, col_item = col_item)
    recoms = predict_u2i(model, xs, n; kwargs...)
    result = metric(recoms, ys)
    return result
end
