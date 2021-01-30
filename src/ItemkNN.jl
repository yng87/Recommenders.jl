@with_kw_noshow mutable struct ItemkNN <: MLJBase.Unsupervised
    k::Int = 10
    shrink::Float64 = 0
    weighting::Union{Nothing, Symbol} = nothing
end

function MLJBase.fit(model::ItemkNN, verbosity::Int, X)
    raw = unpack(X, in((:userid, :itemid, :target)),
        :userid=>Multiclass, :itemid=>Multiclass, :target=>Continuous)
    user2uidx = Dict(uid=>i for (i, uid) in enumerate(unique(raw[:, :userid])))
    item2iidx = Dict(iid=>i for (i, iid) in enumerate(unique(raw[:, :itemid])))

    Xsparse = transform2sparse(raw, user2uidx, item2iidx)
    if model.weighting == :tfidf
        Xsparse = tfidf(Xsparse)
    end
    sparse_similarity_matrix = compute_similarity_matrix(Xsparse, model.k, model.shrink)

    fitresult = (sparse_similarity_matrix, user2uidx, item2iidx)
    report    = nothing
    cache     = nothing
    return fitresult, cache, report
end

function MLJBase.predict(model::ItemkNN, fitresult, Xnew)
    X = unpack(Xnew, in((:userid, :itemid, :target))
        :userid=>Multiclass, :itemid=>Multiclass, :target=>Continuous)
    sparse_similarity_matrix, user2uidx, item2iidx = fitresult
    iidx2item = Dict(i=>iid for (iid, i) in item2iidx)
    Xsparse = transform2sparse(X, user2uidx, item2iidx)

    userids = unique(X[:, :userid])
    preds = []
    for uid in userids
        uidx = user2uidx[uid]
        pred = sortperm((Xsparse[uidx, :]' * sparse_similarity_matrix)', rev=true)

        I, R = findnz(Xsparse[uidx, :])
        filter!(p->!(p in I), pred)

        pred = [iidx2item[i] for i in pred]

        append!(preds, [uid, pred])
    end
end

function transform2sparse(X, user2uidx, item2iidx)
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
    
    return sparse(U, I, R)
end

function tfidf(X::SparseMatrixCSC)
    U, I, R = findnz(X)
    n_users, n_items = size(X)
        
    bincount = zeros(n_users)
    for u in U
        bincount[u] += 1
    end

    idf = log.(n_items ./ (bincount .+ 1e-6)) .+ 1

    for j in 1:length(U)
        R[j] = R[j] * idf[U[j]]
    end

    return sparse(U, I, R)
end

function compute_similarity_matrix(X::SparseMatrixCSC, topK::Int, shrink::Float64)
    # (user, item)^T * (user, item) -> (item, item)
    # Return S[i, j] where j is full items, and i is related items at topK
    
    simJ = Int[]
    simI = Int[]
    simS = Float64[]
    
    U, I, R = findnz(X)
    n_users, n_items = size(X)
    
    norms = sqrt.(sum(X.^2, dims=1))
    
    for j in 1:n_items
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
        arg_sort_i = sortperm(simj, rev=true)[2:topK+1]
        append!(simI, arg_sort_i)
        append!(simS, simj[arg_sort_i])
        append!(simJ, fill(j, length(arg_sort_i)))
    end
    
    return sparse(simI, simJ, simS)
end