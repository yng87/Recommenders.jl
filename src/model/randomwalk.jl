"""
    Randomwalk()

Recommendation model using random walk with restart on user-item bipartite graph. Implemented algorithm is based on Pixie random walk.

# References
C.  Eksombatchai (2018), [Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time](http://dl.acm.org/citation.cfm?doid=3178876.3186183)
"""
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

"""
    fit!(model::Randomwalk, table; col_user = :userid, col_item = :itemid)

Build bipartite graph from `table`. One side of the graph collcets user nodes, and the others item nodes. If a user actions an item, an edge is added between them. The graph is undirected, and has no extra weights.
"""
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

@doc raw"""
    predict_u2i(model::Randomwalk, userid::Union{AbstractString,Int}, n::Int64; drop_history = false, terminate_prob = 0.1, total_walk_length = 10000, min_high_visited_candidates = Inf, high_visited_count_threshold = Inf, pixie_walk_length_scaling = false, pixie_multi_hit_boosting = false, aggregate_function = sum)

Make recommendation by random walk with restart. Basic algorithm is as follows:

1. Get items that are already consumed by the user (on the graph, they are connected by one step). We denote them by ``q \in Q``.
2. Starting from each node ``q \in Q``, perform multiple random walks with certain stop probability. Record the visited count of the items on the walk. We denote the counts of item ``p`` on the walk from ``q`` by ``V_q[p]``.
3. Finally aggregate ``V_q[p]`` to ``V[p]``, and recommeds top-scored items. Two mothods for aggregation are provided
- Simple aggregation: Taking sum, ``V[p] = \sum_{q\in Q} V_q[p]``. You can also replace `sum` by, for instance, `maximum`.
- Pixie boosting: ``V[p] = (\sum_{q\in Q} \sqrt{V_q[p]})^2``, putting more importance on the nodes visited by ``q``s.

# Model-specific arguments
- `terminate_prob`: stop probability of one random walk
- `total_walk_length`: total walk length over the multiple walks from ``q``'s.
- `high_visited_count_threshold`: early stopping paramerer. Count up `high_visited_count` when the visited count of certain node reaches this threshold.
- `min_high_visited_candidates`: early stopping parameter. Terminate the walk from some node ``q`` if `high_visited_count` reaches `min_high_visited_candidates`.
- `pixie_walk_length_scaling`: If set to true, the start node ``q`` with more degree will be given more walk length. If false, the walk length is the same over all the nodes ``q \in Q``
- `pixie_multi_hit_boosting`: If true, pixie boosting is adopted for aggregation. If false, simple aggregation is used.
- `aggregate_function`: function used by simple aggregation.
"""
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
