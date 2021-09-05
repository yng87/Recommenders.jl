mutable struct SLIM <: AbstractRecommender
    l1_coeff::Float64
    l2_coeff::Float64

    w::Any

    user2uidx::Union{Dict,Nothing}
    item2iidx::Union{Dict,Nothing}
    iidx2item::Union{Dict,Nothing}
    user_history::Union{Dict,Nothing}

    SLIM(l1_coeff::Float64, l2_coeff::Float64) =
        new(l1_coeff, l2_coeff, nothing, nothing, nothing, nothing, nothing)
end



function predict(model::SLIM, uidx, iidx)::Float64
    pred = 0.0
    for i in model.user_history[uidx]
        pred += model.w[iidx, i]
    end
    return pred
end


function cd!(model::SLIM, i, Y, inner_prod, squared_norms)
    model.w[i, i] = 0

    # cache common value to optimize inner loop below.
    ks, ws = findnz(model.w[i, :])
    cached_yhat = spzeros(size(Y)[1])
    for (k, w) in zip(ks, ws)
        cached_yhat += w * Y[:, k]
    end

    for j = 1:size(Y)[2]
        if i == j
            continue
        end

        yi_yj = inner_prod[i, j]
        # need to subtranct if w_ij != 0
        yhat_yj = cached_yhat' * Y[:, j] - model.w[i, j] * inner_prod[j, j]
        denom = squared_norms[j] + model.l2_coeff

        current_wij = model.w[i, j]
        if (yi_yj - yhat_yj - model.l1_coeff) > 0 && model.w[i, j] > 0
            model.w[i, j] = (yi_yj - yhat_yj - model.l1_coeff) / denom
        elseif (yi_yj - yhat_yj + model.l1_coeff) < 0 && model.w[i, j] < 0
            model.w[i, j] = (yi_yj - yhat_yj + model.l1_coeff) / denom
        else
            model.w[i, j] = 0
        end
        updated_wij = model.w[i, j]

        if current_wij != updated_wij
            cached_yhat += (updated_wij - current_wij) * Y[:, j]
        end
    end
end

function compute_loss(model::SLIM, i, inner_prod, norms)
    loss = 0
    js, ws = findnz(model.w[i, :])
    for (j, w1) in zip(js, ws)
        for (k, w2) in zip(js, ws)
            loss += 0.5 * w1 * inner_prod[j, k] * w2
        end
    end

    for (j, w) in zip(js, ws)
        loss += -inner_prod[i, j] * w
        loss += 0.5 * model.l2_coeff * w^2
        loss += model.l1_coeff * abs(w)
    end

    loss += 0.5 * norms[i]

    return loss
end

function fit!(
    model::SLIM,
    table;
    callbacks = Any[],
    col_user = :userid,
    col_item = :item_id,
    col_rating = :rating,
    n_epochs = 2,
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

    model.w = sprandn(n_item, n_item, 0.3)
    for i = 1:n_item
        model.w[i, i] = 0
    end

    # callback is any callbale with same interface
    callbacks = append!(Any[LogTrainLoss()], callbacks)
    for cb in callbacks
        if typeof(cb) <: AbstractCallback
            initialize!(cb, model, col_user = col_user, col_item = col_item)
        end
    end

    inner_prod = Y' * Y
    squared_norms = dropdims(sum(Y .^ 2, dims = 1), dims = 1)

    for epoch = 1:n_epochs
        train_losses = Vector{Float64}(undef, n_item)

        Threads.@threads for i = 1:n_item
            cd!(model, i, Y, inner_prod, squared_norms)
            train_losses[i] = compute_loss(model, i, inner_prod, squared_norms)
        end
        train_loss = sum(train_losses)

        try
            for cb in callbacks
                cb(model, train_loss, epoch, verbose)
            end
        catch StopTrain
            break
        end
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
