"""
    AbstractRecommender

Abstract struct for all recommendation models.
"""
abstract type AbstractRecommender end


"""
    fit!(model::AbstractRecommender, table; kwargs...)

Train `model` by `table`. `kwargs` is model-dependent parameters.
`table` is any `Tables.jl`-compatible data.
"""
function fit!(model::AbstractRecommender, table; kwargs...)
    throw("Not implemented.")
end


"""
    predict_u2i(model, userid, n; kwargs...)

Predict items to single user. 

# Arguments
- `model::AbstractRecommender`: trained model.
- `userid::Union{AbstractString,Int}`: userid to get predictions.
- `n::Int64`: number of retrieved items.

# Keyword arguments
- `drop_history::Bool`: whether to drop already consumed items from predictions.

# Return
Vector of predicted items, ordered by descending score.
"""
function predict_u2i(
    model::AbstractRecommender,
    userid::Union{AbstractString,Int},
    n::Int64;
    kwargs...,
)
    throw("Not implemented.")
end


"""
    predict_u2i(model, userid, n; kwargs...)

Predict items to single user. 

# Arguments
- `model::AbstractRecommender`: trained model.
- `userids::Vector{Any}`: userids to get predictions.
- `n::Int64`: number of retrieved items.

# Keyword arguments
- `drop_history::Bool`: whether to drop already consumed items from predictions.

# Return
Vector of predicted items, each element corresponds to list of predictions to the user.
"""
function predict_u2i(model::AbstractRecommender, userids::Vector{Any}, n::Int64; kwargs...)
    recoms = Vector{Vector{Union{AbstractString,Int}}}(undef, length(userids))
    Threads.@threads for i in eachindex(userids)
        userid = userids[i]
        recoms[i] = predict_u2i(model, userid, n; kwargs...)
    end
    return recoms
end


"""
    evaluate_u2i(model, train_table, test_table, metric, n; kwargs...)

Perform fit! `model` on `train_table`, predict for each user in `test_table`, and evaluate by `metric`.

# Arguments
- `model::AbstractRecommender`: model to evaluate.
- `train_table`: any `Tables.jl`-compatible data for train.
- `test_table`: any `Tables.jl`-compatible data for test.
- `metric`: evaluation metrics, `MeanMetric` of list of `MeanMetric`.
- `n::Int64`: number of retrieved items.

# Keyword arguments
- `drop_history::Bool`: whether to drop already consumed items from predictions.
- any model-dependent arguments.

# Return
Evaluated metrics for `test_table`.
"""
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
        result[Symbol(name(metric))] = metric(recoms, gts)
    end
    return NamedTuple(result)
end
