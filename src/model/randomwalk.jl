mutable struct Randomwalk <: AbstractRecommender
    adjacency_list::Union{Vector{Int},Nothing}
    offsets::Union{Vector{Int},Nothing}

    user_history::Any
    user2uidx::Union{Dict,Nothing}
    item2iidx::Union{Dict,Nothing}
    iidx2item::Union{Dict,Nothing}
    max_degree::Union{Int,Nothing}

    Randomwalk() = new(nothing, nothing, nothing, nothing, nothing, nothing, nothing)
end

function fit!(model::Randomwalk, table; col_user = :userid, col_item = :itemid, kwargs...)
    model.adjacency_list, model.offsets, model.user2uidx, model.item2iidx =
        build_graph(table, col_user = col_user, col_item = col_item)

    model.iidx2item = Dict(iidx => item for (item, iidx) in model.item2iidx)

    model.user_history = Dict{Int,Vector{Int}}()
    for (userid, history) in
        zip(make_u2i_dataset(table, col_user = col_user, col_item = col_item)...)
        uidx = model.user2uidx[userid]
        history = [model.item2iidx[item] for item in history]
        model.user_history[uidx] = history
    end

    model.max_degree = get_max_degree(model.offsets)
end

function predict_u2i(
    model::Randomwalk,
    userid::Union{AbstractString,Int},
    n::Int64;
    drop_history = false,
    terminate_prob = 0.1,
    total_walk_length = 10000,
    min_high_visited_candidates = Inf,
    high_visited_count_threshold = Inf,
    pixie_walk_length_scaling = false,
    pixie_multi_hit_boosting = false,
    aggregate_function = sum,
    kwargs...,
)
    if !(userid in keys(model.user2uidx))
        return []
    end

    uidx = model.user2uidx[userid]
    query_nodeids = model.user_history[uidx]

    visited_count = randomwalk_multiple(
        model.adjacency_list,
        model.offsets,
        query_nodeids,
        terminate_prob,
        total_walk_length,
        min_high_visited_candidates,
        high_visited_count_threshold,
        pixie_walk_length_scaling,
        pixie_multi_hit_boosting,
        model.max_degree,
        aggregate_function,
    )

    sorted_idx = sortperm(collect(values(visited_count)), rev = true)
    pred_iidx = collect(keys(visited_count))[sorted_idx]

    # this is very slow
    if drop_history
        filter!(p -> !(p in query_nodeids), pred_iidx)
    end

    pred_items = [model.iidx2item[iidx] for iidx in pred_iidx]

    n = min(n, length(pred_items))
    return pred_items[1:n]
end
