"""
    AbstractRecommender

Abstract struct for all recommendation models.
"""
abstract type AbstractRecommender end


"""
    fit!(model::AbstractRecommender, table; kwargs...)

Train `model` by `table`.

# Arguments
- `model`: concrete type under `AbstractRecommender`
- `table`: any `Tables.jl`-compatible data for train.

# Keyword arguments
- `col_user`: name of user column in `table`
- `col_item`: name of item column in `table`
- and other model-dependent arguments.
"""
function fit!(model::AbstractRecommender, table; kwargs...)
    throw("Not implemented.")
end


"""
    predict_u2i(model, userid, n; kwargs...)

Make recommendations to user (or users). When `userid` is a collection of raw user ids, this function performs parallel predictions by `Threads.@threads`.

# Arguments
- `model::AbstractRecommender`: trained model.
- `userid:`: user id to get predictions. type is `AbstractString`, `Int` or their collection.
- `n::Int64`: number of retrieved items.

# Keyword arguments
- `drop_history::Bool`: whether to drop already consumed items from predictions.
- and other model-dependent arguments.

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

function predict_u2i(
    model::AbstractRecommender,
    userids::Vector{<:Union{AbstractString,Int}},
    n::Int64;
    kwargs...,
)
    recoms = Vector{Vector{Union{AbstractString,Int}}}(undef, length(userids))
    Threads.@threads for i in eachindex(userids)
        userid = userids[i]
        recoms[i] = predict_u2i(model, userid, n; kwargs...)
    end
    return recoms
end

"""
    predict_i2i(model, itemid, n; kwargs...)

Make recommendations given an item. When `itemid` is a collection of raw item ids, this function performs parallel predictions by `Threads.@threads`.

# Arguments
- `model::AbstractRecommender`: trained model.
- `itemid:`: item id to get predictions. type is `AbstractString`, `Int` or their collection.
- `n::Int64`: number of retrieved items.

# Keyword arguments
- other model-dependent arguments.

# Return
Vector of predicted items, ordered by descending score.
"""
function predict_i2i(
    model::AbstractRecommender,
    itemid::Union{AbstractString,Int},
    n::Int64;
    kwargs...,
)
    throw("Not implemented")
end

function predict_i2i(
    model::AbstractRecommender,
    itemids::Vector{<:Union{AbstractString,Int}},
    n::Int64;
    kwargs...,
)
    recoms = Vector{Vector{Union{AbstractString,Int}}}(undef, length(itemids))
    Threads.@threads for i in eachindex(itemids)
        itemid = itemids[i]
        recoms[i] = predict_i2i(model, itemid, n; kwargs...)
    end
    return recoms
end



"""
    evaluate_u2i(model, train_table, test_table, metric, n; kwargs...)

Perform `fit!` for `model` on `train_table`, predict for each user in `test_table`, and evaluate by `metric`.

# Arguments
- `model::AbstractRecommender`: model to evaluate.
- `train_table`: any `Tables.jl`-compatible data for train.
- `test_table`: any `Tables.jl`-compatible data for test.
- `metric`: evaluation metrics, `MeanMetric` or collection of `MeanMetric`.
- `n::Int64`: number of retrieved items.

# Keyword arguments
- `drop_history::Bool`: whether to drop already consumed items from predictions.
- `col_user`: name of user column in `table`
- `col_item`: name of item column in `table`
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

"""
    save_model(model::AbstractRecommender, filepath, overwrite = false)

Save model by JLD2.

# Arguments
- `model::AbstractRecommender`: model to save.
- `filepath`: path to save the model. If the model save multiple files, this argument points to directory.
- `overwrite`: whether to overwrite if `filepath` already exists.
"""
function save_model(model::AbstractRecommender, filepath, overwrite = false)
    if ispath(filepath)
        if overwrite
            rm(filepath; force = true, recursive = true)
        else
            throw("$(filepath) already exists.")
        end
    end
    mkpath(dirname(filepath))
    jldsave(filepath; model = model)
    return filepath
end

"""
    load_model(model::AbstractRecommender, filepath)

Load model by JLD2.

# Arguments
- `filepath`: path from which load the model. If the model save multiple files, this argument points to directory.
"""
function load_model(filepath)
    return load(filepath, "model")
end