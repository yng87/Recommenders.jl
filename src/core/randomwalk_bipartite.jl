get_degree(offsets, i::Int) = offsets[i+1] - offsets[i]
get_degree(offsets, is::Vector{Int}) = [get_degree(offsets, i) for i in is]

function get_max_degree(offsets)
    allnode = collect(1:(length(offsets)-1))
    degrees = get_degree(offsets, allnode)
    return max(degrees...)
end

function get_neighbor(adjacency_list, offsets, nodeid)
    degree = get_degree(offsets, nodeid)
    adjacency_list[offsets[nodeid]:offsets[nodeid]+degree-1]
end

function onewalk(adjacency_list, offsets, query_nodeid)
    degree = get_degree(offsets, query_nodeid)
    if degree == 0
        # isolated node
        return -1
    end
    rand_position = floor(Int, rand(Uniform(0, degree)))
    return adjacency_list[offsets[query_nodeid]+rand_position]
end


function randomwalk(
    adjacency_list,
    offsets,
    query_nodeid,
    count_same_nodetype,
    terminate_prob,
    total_walk_length,
    min_high_visited_candidates,
    high_visited_count_threshold,
)::Dict{Int,Int}
    visited_count = Dict{Int,Int}()
    current_length = 0
    n_high_visited_candidates = 0
    while (current_length < total_walk_length) &&
        (n_high_visited_candidates < min_high_visited_candidates)
        walk_length = rand(Geometric(terminate_prob))
        current_nodeid = query_nodeid

        for _ = 1:walk_length
            current_nodeid = onewalk(adjacency_list, offsets, current_nodeid)
            if count_same_nodetype
                current_nodeid = onewalk(adjacency_list, offsets, current_nodeid)
                # TODO: what if -1 returned?
            end

            if current_nodeid in keys(visited_count)
                visited_count[current_nodeid] += 1
            else
                visited_count[current_nodeid] = 1
            end

            if visited_count[current_nodeid] == high_visited_count_threshold
                n_high_visited_candidates += 1
            end

            if !count_same_nodetype
                current_nodeid = onewalk(adjacency_list, offsets, current_nodeid)
                # TODO: what if -1 returned?
            end
        end

        current_length += walk_length
    end

    return visited_count
end

function pixie_multi_hit_boost(visited_counts::Vector{Dict{Int,Int}})::Dict{Int,Real}
    if length(visited_counts) == 1
        allnodes = keys(visited_counts[1])
    else
        allnodes = union(keys.(visited_counts)...)
    end
    boosted_visited_count = Dict{Int,Real}()
    for nodeid in allnodes
        counts = get.(visited_counts, nodeid, 0)
        boosted_visited_count[nodeid] = (sum(sqrt.(counts)))^2
    end
    return boosted_visited_count
end

function aggregate_multi_randomwalk(
    visited_counts::Vector{Dict{Int,Int}},
    aggregate_function = sum,
)::Dict{Int,Int}
    if length(visited_counts) == 1
        allnodes = keys(visited_counts[1])
    else
        allnodes = union(keys.(visited_counts)...)
    end
    total_visited_count = Dict{Int,Int}()
    for nodeid in allnodes
        total_visited_count[nodeid] = aggregate_function(get.(visited_counts, nodeid, 0))
    end
    return total_visited_count
end

function randomwalk_multiple(
    adjacency_list,
    offsets,
    query_nodeids,
    count_same_nodetype,
    terminate_prob,
    total_walk_length,
    min_high_visited_candidates,
    high_visited_count_threshold,
    pixie_walk_length_scaling = false,
    pixie_multi_hit_boosting = false,
    max_degree = nothing,
    aggregate_function = sum,
)::Dict{Int,Real}

    if max_degree === nothing
        max_degree = get_max_degree(offsets)
    end

    degrees = get_degree(offsets, query_nodeids)

    if pixie_walk_length_scaling
        scaling_factors = degrees .* (max_degree .- log.(degrees))
        scaling_factors = scaling_factors / sum(scaling_factors)
    else
        scaling_factors = ones(length(query_nodeids)) / length(query_nodeids)
    end

    @assert length(query_nodeids) == length(scaling_factors)

    visited_counts = Vector{Dict{Int,Int}}(undef, length(query_nodeids))

    for (i, query_nodeid) in enumerate(query_nodeids)
        this_total_walk_length = round(Int, total_walk_length * scaling_factors[i])
        visited_counts[i] = randomwalk(
            adjacency_list,
            offsets,
            query_nodeid,
            count_same_nodetype,
            terminate_prob,
            this_total_walk_length,
            min_high_visited_candidates,
            high_visited_count_threshold,
        )
    end

    if pixie_multi_hit_boosting
        return pixie_multi_hit_boost(visited_counts)
    else
        return aggregate_multi_randomwalk(visited_counts, aggregate_function)
    end
end

function build_graph(table; col_user = :userid, col_item = :itemid)
    table, user2uidx, item2iidx, _ =
        make_idmap(table, col_user = col_user, col_item = col_item)

    n_user = length(user2uidx)
    n_item = length(item2iidx)

    df = DataFrame(table)

    # nodeid = 1 - n_item: item node
    # nodeid = (n_item + 1) - (n_item + n_user): user node
    df[!, col_user] = df[!, col_user] .+ n_item
    for user in keys(user2uidx)
        user2uidx[user] = user2uidx[user] + n_item
    end

    adjacency_list = sort(df, [col_item])[!, col_user]
    append!(adjacency_list, sort(df, [col_user])[!, col_item])

    item_degrees = sort(combine(groupby(df, col_item), nrow), [col_item])[!, :nrow]
    user_degrees = sort(combine(groupby(df, col_user), nrow), [col_user])[!, :nrow]

    @assert length(adjacency_list) == sum(item_degrees) + sum(user_degrees)

    offsets = Vector{Int}(undef, n_item + n_user + 1)
    item_cumsum = cumsum(item_degrees) .+ 1
    user_cumsum = cumsum(user_degrees) .+ item_cumsum[end]
    offsets[1] = 1
    offsets[2:(n_item+1)] = item_cumsum
    offsets[(n_item+2):end] = user_cumsum

    return adjacency_list, offsets, user2uidx, item2iidx
end