get_degree(offsets, i::Int) = offsets[i+1] - offsets[i]
get_degree(offsets, is::Vector{Int}) = [get_degree(offsets, i) for i in is]

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
            current_nodeid = onewalk(adjacency_list, offsets, current_nodeid)
            # TODO: what if -1 returned?

            if current_nodeid in keys(visited_count)
                visited_count[current_nodeid] += 1
            else
                visited_count[current_nodeid] = 1
            end

            if visited_count[current_nodeid] == high_visited_count_threshold
                n_high_visited_candidates += 1
            end
        end

        current_length += walk_length
    end

    return visited_count
end

function pixie_multi_hit_boost(visited_counts::Vector{Dict{Int,Int}})::Dict{Int,Int}
    allnodes = union(keys.(visited_counts)...)
    boosted_visited_count = Dict(Int, Int)()
    for nodeid in allnodes
        counts = get.(visited_counts, nodeid, 0)
        boosted_visited_count[nodeid] = (sum(sqrt.(counts)))^2
    end
    return boosted_visited_count
end

function sum_multi_randomwalk_count(visited_counts::Vector{Dict{Int,Int}})::Dict{Int,Int}
    total_visited_count::Dict(Int, Int) = visited_counts[1]
    for i = 2:length(visited_counts)
        for (nodeid, n) in visited_counts[i]
            if nodeid in keys(total_visited_count)
                total_visited_count[nodeid] += n
            else
                total_visited_count[nodeid] = n
            end
        end
    end
    return total_visited_count
end

function get_max_degree(offsets)
    allnode = 1:(length(offsets)-1)
    degrees = get_degree(offsets, allnode)
    return max(degrees...)
end


function randomwalk_multiple(
    adjacency_list,
    offsets,
    query_nodeids,
    terminate_prob,
    total_walk_length,
    min_high_visited_candidates,
    high_visited_count_threshold,
    max_degree = nothing,
    pixie_multi_hit_boosting = true,
)::Dict{Int,Int}

    if max_degree === nothing
        max_degree = get_max_degree(offsets)
    end

    degrees = get_degree(offsets, query_nodeids)
    scaling_factors = degrees .* (max_degree .- log.(degrees))
    scaling_factors = scaling_factors / sum(scaling_factors)

    visited_counts = Vector{Dict{Int,Int}}(undef, length(query_nodeids))

    for (i, query_nodeid) in enumerate(query_nodeids)
        this_total_walk_length = round(Int, total_walk_length * scaling_factors[i])
        visited_counts[i] = randomwalk(
            adjacency_list,
            offsets,
            query_nodeid,
            terminate_prob,
            this_total_walk_length,
            min_high_visited_candidates,
            high_visited_count_threshold,
        )
    end

    if pixie_multi_hit_boosting
        return pixie_multi_hit_boost(visited_counts)
    else
        return sum_multi_randomwalk_count(visited_counts)
    end
end
