mutable struct SLIM <: AbstractRecommender
    l1_ratio::Float64
    λminratio::Float64
    k::Int

    W

    user2uidx::Union{Dict,Nothing}
    item2iidx::Union{Dict,Nothing}
    iidx2item::Union{Dict,Nothing}
    R

    SLIM(l1_ratio::Float64=0.5, λminratio::Float64=1e-4, k::Int=-1) =
        new(l1_ratio, λminratio, k, nothing, nothing, nothing, nothing, nothing)
end

function truncate_at_k!(w::SparseVector, k::Int)
    is, ws = findnz(w)
    if k>=length(ws) || k<1
        return
    end
    arg_outof_k = sortperm(ws, rev=true)[(k+1):end]
    w[is[arg_outof_k]] .= 0
end

function fit!(
    model::SLIM,
    table;
    col_user = :userid,
    col_item = :itemid,
    col_rating = :rating,
    cd_tol=1e-7, 
    nλ=100,
    kwargs...,
)
    table, model.user2uidx, model.item2iidx, model.iidx2item =
        make_idmap(table, col_user = col_user, col_item = col_item)

    model.R = rows2sparse(
        table,
        col_user = col_user,
        col_item = col_item,
        col_rating = col_rating,
    )

    unique_items = collect(keys(model.iidx2item))
    n_item = length(unique_items)

    model.W=spzeros(n_item, n_item)
    # for multithreading
    tmp_ws = Vector{SparseVector{Float64, Int}}(undef, n_item)

    Threads.@threads for i = 1:n_item
        mask = spzeros(n_item, n_item)
        mask[i, i] = 1
        lasso_res = fit(LassoPath, model.R - model.R*mask, Vector(model.R[:, i]), α=model.l1_ratio, λminratio=model.λminratio, standardize=false, intercept=false, stopearly=true, cd_tol=cd_tol, nλ=nλ)
        wi = lasso_res.coefs[:, end]
        truncate_at_k!(wi, model.k)
        dropzeros!(wi)
        tmp_ws[i] = wi
    end

    for i in 1:n_item
        model.W[:, i] = tmp_ws[i]
    end
end

function predict_u2i(
    model::SLIM,
    userid::Union{AbstractString,Int},
    n::Int64;
    drop_history = false,
    kwargs...,
)
    if !(userid in keys(model.user2uidx))
        return []
    end

    uidx = model.user2uidx[userid]
    user_history = model.R[uidx, :]

    pred = (user_history'*model.W)' 
    viewed_item, ratings = findnz(user_history)
    for (iidx, rui) in zip(viewed_item, ratings)
        pred[iidx] -= rui * model.W[iidx, iidx]
    end

    pred_iidx = sortperm(pred, rev=true)
    # this is very slow
    if drop_history
        filter!(p -> !(p in viewed_item), pred_iidx)
    end

    pred_items = [model.iidx2item[iidx] for iidx in pred_iidx]

    n = min(n, length(pred_items))
    return pred_items[1:n]
end
