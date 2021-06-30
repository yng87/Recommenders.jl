"""
    kNNRecommender

- k: compute k nearest neighbor similarity.
- shrink: if nonzero, decrease contributions from items with few rating.
- weighting: currently supoorts TF-IDF
- npred: number of retrieval by predict.
"""
@with_kw_noshow mutable struct kNNRecommender <: BaseRecommender
    # config
    k::Int = 100
    shrink::Float64 = 0
    weighting::Union{Nothing,Symbol} = nothing
    # learned params
    similarity = nothing
end

"""
    fit(model::kNNRecommender, inter)

fit `kNNRecommender`. `inter` is interaction matrix.
"""
function fit!(model::kNNRecommender, inter::SparseMatrixCSC)
    if model.weighting == :tfidf
        inter = tfidf(inter)
    end
    model.similarity = compute_similarity(inter, model.k, model.shrink)
    return model
end

function retrieve(model::kNNRecommender, user_history, n; drop_history=false)
    similarity = model.similarity
    pred = sortperm(similarity * user_history, rev = true)

    if drop_history
        filter!(p -> !(p in user_history), pred)
    end
    n = min(n, length(pred))
    return pred[1:n]
end

function tfidf(X::SparseMatrixCSC)
    U, I, R = findnz(X)
    n_users, n_items = size(X)

    bincount = zeros(n_users)
    for u in U
        bincount[u] += 1
    end

    idf = log.(n_items ./ (bincount .+ 1e-6)) .+ 1

    for j = 1:length(U)
        R[j] = R[j] * idf[U[j]]
    end

    return sparse(U, I, R)
end

function compute_similarity(X::SparseMatrixCSC, topK::Int, shrink::Float64)
    # (user, item)^T * (user, item) -> (item, item)
    # Return S[i, j] where j is full items, and i is related items at topK

    simJ = Int[]
    simI = Int[]
    simS = Float64[]

    U, I, R = findnz(X)
    n_users, n_items = size(X)

    norms = sqrt.(sum(X .^ 2, dims = 1))

    for j = 1:n_items
        Uj, Rj = findnz(X[:, j])
        simj = zeros(n_items)
        for (u, ruj) in zip(Uj, Rj)
            Iu, Ri = findnz(X[u, :])
            for (i, rui) in zip(Iu, Ri)
                s = rui * ruj
                s /= norms[j] * norms[i] + shrink + 1e-6
                simj[i] += s
            end
        end
        arg_sort_i = sortperm(simj, rev = true)[2:topK+1]
        append!(simI, arg_sort_i)
        append!(simS, simj[arg_sort_i])
        append!(simJ, fill(j, length(arg_sort_i)))
    end

    return sparse(simI, simJ, simS)
end