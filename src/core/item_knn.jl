function tfidf(X::SparseMatrixCSC)
    # item = document
    # user = term
    U, I, R = findnz(X)
    n_users, n_items = size(X)

    bincount = zeros(n_users)
    for u in U
        bincount[u] += 1
    end

    idf = log.(n_items ./ (bincount .+ 1))

    for j = 1:length(U)
        R[j] = R[j] * idf[U[j]]
    end

    return sparse(U, I, R)
end

function bm25(X::SparseMatrixCSC, k1 = 1.2, b = 0.75)
    U, I, R = findnz(X)
    n_users, n_items = size(X)

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

    return sparse(U, I, R)
end

function compute_similarity(
    X::SparseMatrixCSC,
    topK::Int,
    shrink::Float64,
    normalize::Bool,
    normalize_similarity::Bool,
)
    # (user, item)^T * (user, item) -> (item, item)
    # Return S[i, j] where j is full items, and i is related items at topK
    if shrink < 0
        throw(ArgumentError("shrink must be 0 or positive."))
    end

    @info "Prepare"
    n_users, n_items = size(X)
    topK = min(topK, n_items - 1)

    simJ = Vector{Int64}(undef, topK * n_items)
    simI = Vector{Int64}(undef, topK * n_items)
    simS = Vector{Float64}(undef, topK * n_items)

    @info "Compute norm"
    norms = sqrt.(sum(X .^ 2, dims = 1))
    norms = dropdims(norms, dims = 1)

    @info "Make cache"
    # to speed up, cache non zero indices
    nonzero_I_R = Vector{Tuple{Vector{Int},Vector{<:Real}}}(undef, n_users)
    Threads.@threads for u = 1:n_users
        nonzero_I_R[u] = findnz(X[u, :])
    end

    @info "Compute similarity by multithreading"
    Threads.@threads for j = 1:n_items
        Uj, Rj = findnz(X[:, j])
        simj = zeros(n_items)
        for (u, ruj) in zip(Uj, Rj)
            Iu, Ri = nonzero_I_R[u]
            for (i, rui) in zip(Iu, Ri)
                s = rui * ruj
                if normalize
                    s /= norms[j] * norms[i] + shrink + 1e-6
                end
                simj[i] += s
            end
        end
        arg_sort_i = sortperm(simj, rev = true)[2:topK+1]
        simI[(1+(j-1)*topK):j*topK] = arg_sort_i
        simS[(1+(j-1)*topK):j*topK] = simj[arg_sort_i]
        simJ[(1+(j-1)*topK):j*topK] = fill(j, length(arg_sort_i))
    end

    similarity = sparse(simI, simJ, simS)

    if normalize_similarity
        # see M. Deshpande and G. Karypis (2004)
        # https://doi.org/10.1145/963770.963776
        for i = 1:size(similarity)[2]
            similarity[:, i] /= sum(similarity[:, i])
        end
    end
    return similarity
end

function predict_u2i(
    similarity::SparseMatrixCSC,
    user_history::Vector,
    n::Int64;
    drop_history::Bool = false,
)
    user_history = sparse(user_history)
    return predict_u2i(similarity, user_history, n; drop_history = drop_history)
end


function predict_u2i(
    similarity::SparseMatrixCSC,
    user_history::SparseVector,
    n::Int64;
    drop_history::Bool = false,
)
    pred_iidx, pred_score = findnz(similarity * user_history)
    permsocre = sortperm(pred_score, rev = true)
    pred = pred_iidx[permsocre]

    viewed_item, _ = findnz(user_history)
    # this is very slow
    if drop_history
        pred = filter(p -> !(p in viewed_item), pred)
    end
    n = min(n, length(pred))
    return pred[1:n]
end

function predict_i2i(similarity::SparseMatrixCSC, iidx::Int, n::Int)
    pred_iidx, pred_score = findnz(similarity[:, iidx])
    n = min(n, length(pred_iidx))
    return pred_iidx[sortperm(pred_score, rev = true)][1:n]
end