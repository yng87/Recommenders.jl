"""
    ItemkNN

- k: compute k nearest neighbor similarity.
- shrink: if nonzero, decrease contributions from items with few rating.
- weighting: currently supoorts TF-IDF
- npred: number of retrieval by predict.
"""
@with_kw_noshow mutable struct ItemkNN <: MMI.Unsupervised
    # config
    k::Int = 100
    shrink::Float64 = 0
    normalize::Bool = true
    weighting::Union{Nothing,Symbol} = nothing
    col_user = :userid
    col_item = :itemid
    col_rating = :rating
    # learned params
    # similarity = nothing
end

function MMI.fit(model::ItemkNN, verbosity, X)
    X = rows2sparse(
        X,
        col_user = model.col_user,
        col_item = model.col_item,
        col_rating = model.col_rating,
    )
    if model.weighting == :tfidf
        X = tfidf(X)
    end
    similarity = compute_similarity(X, model.k, model.shrink, model.normalize)

    fitresult = (similarity,)
    cache = nothing
    report = nothing
    return fitresult, cache, report
end

function rows2sparse(X; col_user = :userid, col_item = :itemid, col_rating = :rating)
    U = Int[]
    I = Int[]
    R = Float64[]

    for row in Tables.rows(X)
        push!(U, row[col_user])
        push!(I, row[col_item])
        push!(R, row[col_rating])
    end

    return sparse(U, I, R)
end

function retrieve(model::ItemkNN, fitresult, user_history, n; drop_history = false)
    similarity = fitresult[1]
    num = similarity * user_history
    denom = sum(similarity, dims = 2)
    denom = dropdims(denom, dims = 2)
    pred = sortperm(num ./ denom, rev = true)

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

function compute_similarity(X::SparseMatrixCSC, topK::Int, shrink::Float64, normalize::Bool)
    # (user, item)^T * (user, item) -> (item, item)
    # Return S[i, j] where j is full items, and i is related items at topK
    if shrink < 0
        throw(ArgumentError("shrink must be 0 or positive."))
    end

    simJ = Int[]
    simI = Int[]
    simS = Float64[]

    _, n_items = size(X)

    topK = min(topK, n_items - 1)

    norms = sqrt.(sum(X .^ 2, dims = 1))
    norms = dropdims(norms, dims = 1)

    for j = 1:n_items
        Uj, Rj = findnz(X[:, j])
        simj = zeros(n_items)
        for (u, ruj) in zip(Uj, Rj)
            Iu, Ri = findnz(X[u, :])
            for (i, rui) in zip(Iu, Ri)
                s = rui * ruj
                if normalize
                    s /= norms[j] * norms[i] + shrink + 1e-6
                end
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