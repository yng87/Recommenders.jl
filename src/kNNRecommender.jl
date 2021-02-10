"""
    kNNRecommender

- k: compute k nearest neighbor similarity.
- shrink: if nonzero, decrease contributions from items with few rating.
- weighting: currently supoorts TF-IDF
- npred: number of retrieval by predict.
"""
@with_kw_noshow mutable struct kNNRecommender <: MLJBase.Deterministic
    k::Int = 100
    shrink::Float64 = 0
    weighting::Union{Nothing,Symbol} = nothing
    npred::Int = 10
end

"""
    fit(model::kNNRecommender, verbosity::Int, X, y, w=nothing)

fit `kNNRecommender`. For ItemkNN, `X` = userids, `y` = itemids.
`w` is additional weighting, for instance, rating of user to item.
"""
function MLJBase.fit(model::kNNRecommender, verbosity::Int, X, y, w = nothing)
    if isnothing(w)
        if verbosity > 0
            println("w is not specified. It is filled by unity.")
        end
        w = ones(length(y))
    end

    raw = (userid = X, itemid = y, target = w)

    Xsparse, user2uidx, item2iidx = transform2sparse(raw)
    iidx2item = Dict(i => iid for (iid, i) in item2iidx)

    if verbosity > 0 && !isnothing(model.weighting)
        println("Weighting by $(model.weighting).")
    end
    if model.weighting == :tfidf
        Xsparse = tfidf(Xsparse)
    end
    similarity = compute_similarity(Xsparse, model.k, model.shrink)

    fitresult = (similarity, Xsparse, user2uidx, item2iidx, iidx2item)
    report = nothing
    cache = nothing
    return fitresult, cache, report
end

function MLJBase.predict(model::kNNRecommender, fitresult, X)
    n = model.npred
    similarity, rating, user2uidx, item2iidx, iidx2item = fitresult
    preds = Array{Any}(undef, length(X))
    Threads.@threads for i in axes(X, 1)
        uid = X[i]
        if uid in keys(user2uidx)
            uidx = user2uidx[uid]
            # Note: この計算合ってる？
            pred = sortperm(similarity * rating[uidx, :], rev = true)

            # filter out already consumed items.
            I, R = findnz(rating[uidx, :])
            filter!(p -> !(p in I), pred)

            if length(pred) == 0
                pred = nothing
            else
                pred = [iidx2item[i] for i in pred]
                if length(pred) > n
                    pred = pred[1:n]
                end
            end
            preds[i] = pred
        else
            preds[i] = nothing
        end
    end
    return preds
end

function transform2sparse(X)
    user2uidx = Dict(uid => i for (i, uid) in enumerate(unique(X.userid)))
    item2iidx = Dict(iid => i for (i, iid) in enumerate(unique(X.itemid)))

    U = Int[]
    I = Int[]
    R = Float64[]
    rows = Tables.rows(X)

    for row in rows
        uid = row.userid
        iid = row.itemid
        t = row.target

        push!(U, user2uidx[uid])
        push!(I, item2iidx[iid])
        push!(R, t)
    end

    return sparse(U, I, R), user2uidx, item2iidx
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