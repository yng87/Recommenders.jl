function tfidf(
    table::Tables.DictColumnTable;
    col_user = :userid,
    col_item = :itemid,
    col_rating = :rating,
)
    # item = document
    # user = term
    U, I, R = table[col_user], table[col_item], table[col_rating]
    n_users = length(unique(U))
    n_items = length(unique(I))

    bincount = zeros(n_users)
    for u in U
        bincount[u] += 1
    end

    idf = log.(n_items ./ (bincount .+ 1))

    for j = 1:length(U)
        R[j] = R[j] * idf[U[j]]
    end

    return Tables.dictcolumntable(Dict(col_user => U, col_item => I, col_rating => R))
end

function bm25(
    table::Tables.DictColumnTable,
    k1 = 1.2,
    b = 0.75;
    col_user = :userid,
    col_item = :itemid,
    col_rating = :rating,
)
    U, I, R = table[col_user], table[col_item], table[col_rating]
    n_users = length(unique(U))
    n_items = length(unique(I))

    bincount = zeros(n_users)
    for u in U
        bincount[u] += 1
    end

    idf = log.((n_items .- bincount .+ 0.5) ./ (bincount .+ 0.5))

    dl = zeros(n_items)
    for i in I
        dl[i] += 1
    end
    avgdl = sum(dl) / n_items

    for j = 1:length(U)
        R[j] = R[j] * idf[U[j]] * (k1 + 1) / (1 + k1 * (1 - b + b * dl[I[j]] / avgdl))
    end

    return Tables.dictcolumntable(Dict(col_user => U, col_item => I, col_rating => R))
end

function get_rating_history(
    table::Tables.DictColumnTable;
    col_user = :userid,
    col_item = :itemid,
    col_rating = :rating,
)
    uidx2rated_itmes = Dict{Int,Vector{Int}}()
    iidx2rated_users = Dict{Int,Vector{Int}}()
    uidx2rating = Dict{Int,Vector{Float64}}()
    iidx2rating = Dict{Int,Vector{Float64}}()
    for row in Tables.rows(table)
        uidx = row[col_user]
        iidx = row[col_item]
        r = row[col_rating]
        if haskey(uidx2rated_itmes, uidx)
            push!(uidx2rated_itmes[uidx], iidx)
            push!(uidx2rating[uidx], r)
        else
            uidx2rated_itmes[uidx] = [iidx]
            uidx2rating[uidx] = [r]
        end
        if haskey(iidx2rated_users, iidx)
            push!(iidx2rated_users[iidx], uidx)
            push!(iidx2rating[iidx], r)
        else
            iidx2rated_users[iidx] = [uidx]
            iidx2rating[iidx] = [r]
        end
    end

    return uidx2rated_itmes, iidx2rated_users, uidx2rating, iidx2rating
end

function compute_similarity(
    uidx2rated_itmes::Dict{Int,Vector{Int}},
    iidx2rated_users::Dict{Int,Vector{Int}},
    uidx2rating::Dict{Int,Vector{Float64}},
    iidx2rating::Dict{Int,Vector{Float64}},
    topK::Int,
    shrink::Float64,
    normalize::Bool,
    normalize_similarity::Bool,
    include_self::Bool = true,
)
    # (user, item)^T * (user, item) -> (item, item)
    # Return S[i, j] where j is full items, and i is related items at topK
    if shrink < 0
        throw(ArgumentError("shrink must be 0 or positive."))
    end

    n_items = length(keys(iidx2rated_users))

    topK = min(topK, n_items - 1)

    simJ = Vector{Int64}(undef, topK * n_items)
    simI = Vector{Int64}(undef, topK * n_items)
    simS = Vector{Float64}(undef, topK * n_items)

    @debug "Computing norms of rating matrix..."
    norms = zeros(n_items)
    for iidx in keys(iidx2rating)
        norms[iidx] += sqrt(sum(iidx2rating[iidx] .^ 2))
    end

    p = Progress(n_items, 1, "Computing similarity...")
    Threads.@threads for j = 1:n_items
        Uj = iidx2rated_users[j] # users who rated item j
        Rj = iidx2rating[j] # its ratings
        simj = zeros(n_items)
        for (u, ruj) in zip(Uj, Rj)
            Iu = uidx2rated_itmes[u] # items rated by user u
            Ri = uidx2rating[u] # its ratings
            for (i, rui) in zip(Iu, Ri)
                s = rui * ruj
                if normalize
                    s /= norms[j] * norms[i] + shrink + 1e-6
                end
                simj[i] += s
            end
        end
        if include_self
            arg_sort_i = sortperm(simj, rev = true)[1:topK]
        else
            arg_sort_i = sortperm(simj, rev = true)[2:topK+1]
        end
        if normalize_similarity
            # see M. Deshpande and G. Karypis (2004)
            # https://doi.org/10.1145/963770.963776
            simj[arg_sort_i] /= sum(simj[arg_sort_i])
        end
        simI[(1+(j-1)*topK):j*topK] = arg_sort_i
        simS[(1+(j-1)*topK):j*topK] = simj[arg_sort_i]
        # simJ[(1+(j-1)*topK):j*topK] = fill(j, length(arg_sort_i))

        next!(p)
    end

    return simI, simS
end

function predict_u2i(
    similar_items::Vector{Int},
    similarity_scores::Vector{Float64},
    user_rated_itmes::Vector{Int},
    topk::Int,
    n::Int64;
    drop_history::Bool = false,
)
    d = Dict()
    for j in user_rated_itmes
        similar_to_j = (1+(j-1)*topk):j*topk
        for (iidx, score) in
            zip(similar_items[similar_to_j], similarity_scores[similar_to_j])
            if drop_history && j == iidx
                continue
            end
            if haskey(d, iidx)
                d[iidx] += score
            else
                d[iidx] = score
            end
        end
    end
    preds = collect(keys(d))
    scores = collect(values(d))
    n = min(n, length(preds))
    return preds[sortperm(scores, rev = true)][1:n]
end

function predict_i2i(sorted_similar_items::Vector{Int}, iidx::Int, topk::Int, n::Int)
    similar_to_j = (1+(iidx-1)*topk):iidx*topk
    preds = sorted_similar_items[similar_to_j]
    n = min(n, length(preds))
    return preds[1:n]
end
