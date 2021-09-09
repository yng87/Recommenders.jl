mutable struct SLIM <: AbstractRecommender
    l1_ratio::Float64
    λminratio::Float64
    k::Int

    w::Union{Vector{SparseVector{Float64, Int64}}, Nothing}

    user2uidx::Union{Dict,Nothing}
    item2iidx::Union{Dict,Nothing}
    iidx2item::Union{Dict,Nothing}
    user_history::Union{Dict,Nothing}

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

function predict(model::SLIM, uidx, iidx)::Float64
    pred = 0.0
    for i in model.user_history[uidx]
        if i != iidx
            pred += model.w[iidx][i]
        end
    end
    return pred
end

function fit!(
    model::SLIM,
    table;
    col_user = :userid,
    col_item = :item_id,
    col_rating = :rating,
    random=false,
    max_iter = 2,
    tol=1e-4,
    verbose = -1,
    kwargs...,
)
    table, model.user2uidx, model.item2iidx, model.iidx2item =
        make_idmap(table, col_user = col_user, col_item = col_item)

    Y = rows2sparse(
        table,
        col_user = col_user,
        col_item = col_item,
        col_rating = col_rating,
    )

    model.user_history = Dict()
    for uidx = 1:size(Y)[1]
        Is, _ = findnz(Y[uidx, :])
        model.user_history[uidx] = Is
    end

    unique_items = collect(keys(model.iidx2item))
    n_item = length(unique_items)

    model.w = Vector{SparseVector{Float64, Int64}}(undef, n_item)

    Threads.@threads for i = 1:n_item
        mask = spzeros(n_item, n_item)
        mask[i, i] = 1
        lasso_res = fit(LassoPath, Y - Y*mask, Vector(Y[:, i]), α=model.l1_ratio, λminratio=model.λminratio, standardize=false, intercept=false)
        model.w[i] = lasso_res.coefs[:, end]
        truncate_at_k!(model.w[i], model.k)
        dropzeros!(model.w[i])
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
    unique_iidx = collect(keys(model.iidx2item))
    preds = [predict(model, uidx, iidx) for iidx in unique_iidx]
    pred_iidx = unique_iidx[sortperm(preds, rev = true)]
    pred_items = [model.iidx2item[iidx] for iidx in pred_iidx]
    if drop_history
        filter!(e -> !(e in model.user_history[uidx]), pred_items)
    end
    n = min(n, length(pred_items))
    return pred_items[1:n]
end
