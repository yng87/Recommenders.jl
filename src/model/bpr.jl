mutable struct BPR <: AbstractRecommender
    dim::Int64
    loss::LossFunction
    reg_coeff::Float64

    user_embedding::Union{Nothing,Matrix{Float64}}
    item_embedding::Union{Nothing,Matrix{Float64}}

    user2uidx::Union{Dict,Nothing}
    item2iidx::Union{Dict,Nothing}
    iidx2item::Union{Dict,Nothing}
    user_history::Union{Dict,Nothing}

    BPR(dim::Int64, reg_coeff::Float64) = new(
        dim,
        BPRLoss(), # default
        reg_coeff,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end

function predict(model::BPR, uidx, iidx)::Float64
    return model.user_embedding[:, uidx]' * model.item_embedding[:, iidx]
end

function predict(model::BPR, uidx, iidx, jidx)::Float64
    return predict(model, uidx, iidx) - predict(model, uidx, jidx)
end


function sgd!(model::BPR, uidx, iidx, jidx, grad_value, lr)
    reg = model.reg_coeff

    uvec = @view model.user_embedding[:, uidx]
    ivec = @view model.item_embedding[:, iidx]
    jvec = @view model.item_embedding[:, jidx]

    model.user_embedding[:, uidx] -= lr * (grad_value * (ivec - jvec) + reg * uvec)
    model.item_embedding[:, iidx] -= lr * (grad_value * uvec + reg * ivec)
    model.item_embedding[:, jidx] -= lr * (grad_value * (-uvec) + reg * ivec)
end



function fit!(
    model::BPR,
    table;
    col_user = :userid,
    col_item = :item_id,
    learning_rate = 0.01,
    tolerance = 1e-3,
    verbose = -1,
    kwargs...,
)
    model.user_history = Dict()
    for (userid, history) in
        zip(make_u2i_dataset(table, col_user = col_user, col_item = col_item)...)
        model.user_history[userid] = history
    end

    table, model.user2uidx = reindex_id_column(table, col_user)
    table, model.item2iidx = reindex_id_column(table, col_item)
    model.iidx2item = Dict(iidx => itemid for (itemid, iidx) in model.item2iidx)

    n_user = length(keys(model.user2uidx))

    unique_items = collect(keys(model.iidx2item))
    n_item = length(unique_items)

    model.user_embedding = rand(model.dim, n_user)
    model.item_embedding = rand(model.dim, n_item)

    user_column = Tables.getcolumn(table, col_user)
    item_column = Tables.getcolumn(table, col_item)
    pos_inter_idx = 1:length(user_column)
    @show pos_inter_idx

    # initial loss
    pred = predict(model, 1, 1, 2)
    train_loss = model.loss(pred)
    n_sample = 1
    rel_delta = 1e20

    # LearnBPR
    while rel_delta > tolerance
        i = rand(pos_inter_idx)
        uidx = user_column[i]
        iidx = item_column[i]

        jidx = rand(unique_items)
        while jidx in model.user_history[uidx]
            jidx = rand(unique_items)
        end

        pred = predict(model, uidx, iidx, jidx)
        grad_value = grad(model.loss, pred)
        sgd!(model, uidx, iidx, jidx, grad_value, learning_rate)

        n_sample += 1
        pred = predict(model, uidx, iidx, jidx)
        delta = (model.loss(pred) - train_loss) / n_sample
        train_loss += delta

        rel_delta = abs(delta) / train_loss

        if verbose >= 1 && (n_sample % verbose == 0)
            @info "n_sample=$n_sample: train_loss=$train_loss, delta=$delta, cond=$(abs(delta) > tolerance)"
        end
    end
end

function predict_u2i(
    model::BPR,
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
        filter!(e -> !(e in model.user_history[userid]), pred_items)
    end
    n = min(n, length(pred_items))
    return pred_items[1:n]
end
