abstract type AbstractRecommender end

function fit!(model::AbstractRecommender, table; kwargs...)
    throw("Not implemented.")
end

function predict_u2i(
    model::AbstractRecommender,
    userid::Union{AbstractString,Int},
    n::Int64;
    kwargs...,
)
    throw("Not implemented.")
end

function predict_u2i(model::AbstractRecommender, userids::Vector{Any}, n::Int64; kwargs...)
    recoms = Vector{Vector{Union{AbstractString,Int}}}(undef, length(userids))
    Threads.@threads for i in eachindex(userids)
        userid = userids[i]
        recoms[i] = predict_u2i(model, userid, n; kwargs...)
    end
    return recoms
end

function evaluate_u2i(
    model::AbstractRecommender,
    train_table,
    test_table,
    metric::MeanMetric,
    n::Int64;
    kwargs...,
)
    col_user = get(kwargs, :col_user, :userid)
    col_item = get(kwargs, :col_item, :itemid)
    fit!(model, train_table; kwargs...)
    userids, gts = make_u2i_dataset(test_table, col_user = col_user, col_item = col_item)
    recoms = predict_u2i(model, userids, n; kwargs...)
    result = metric(recoms, gts)
    return (; Symbol(metric) => result)
end

function evaluate_u2i(
    model::AbstractRecommender,
    train_table,
    test_table,
    metrics::Union{Vector{<:MeanMetric},Tuple{<:MeanMetric}},
    n::Int64;
    kwargs...,
)
    col_user = get(kwargs, :col_user, :userid)
    col_item = get(kwargs, :col_item, :itemid)
    fit!(model, train_table; kwargs...)
    userids, gts = make_u2i_dataset(test_table, col_user = col_user, col_item = col_item)
    recoms = predict_u2i(model, userids, n; kwargs...)

    result = Dict()
    for metric in metrics
        result[get_name(metric)] = metric(recoms, gts)
    end
    return NamedTuple(result)
end