mutable struct ImplicitMF <: AbstractRecommender
    dim::Int64
    loss::LossFunction
    use_bias::Bool
    reg_coeff::Float64

    μ::Union{Nothing,Float64}
    user_bias::Union{Nothing,Vector{Float64}}
    item_bias::Union{Nothing,Vector{Float64}}
    user_embedding::Union{Nothing,Matrix{Float64}}
    item_embedding::Union{Nothing,Matrix{Float64}}

    user2uidx::Union{Dict,Nothing}
    item2iidx::Union{Dict,Nothing}
    iidx2item::Union{Dict,Nothing}
    user_history::Union{Dict,Nothing}

    ImplicitMF(dim::Int64, use_bias::Bool, reg_coeff::Float64) = new(
        dim,
        Logloss(), # default
        use_bias,
        reg_coeff,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
end



function predict(model::ImplicitMF, uidx, iidx)::Float64
    pred = model.user_embedding[:, uidx]' * model.item_embedding[:, iidx]
    if model.use_bias
        pred += model.μ + model.user_bias[uidx] + model.item_bias[iidx]
    end
    return pred
end


function sgd!(model::ImplicitMF, uidx, iidx, grad_value, lr)
    reg = model.reg_coeff

    if model.use_bias
        model.user_bias[uidx] -= lr * (grad_value + reg * model.user_bias[uidx])
        model.item_bias[iidx] -= lr * (grad_value + reg * model.item_bias[iidx])
        model.μ -= lr * (grad_value + reg * model.μ)
    end
    uvec = @view model.user_embedding[:, uidx]
    ivec = @view model.item_embedding[:, iidx]
    model.user_embedding[:, uidx] -= lr * (grad_value * ivec + reg * uvec)
    model.item_embedding[:, iidx] -= lr * (grad_value * uvec + reg * ivec)
end



function fit!(
    model::ImplicitMF,
    table;
    callbacks = Any[],
    col_user = :userid,
    col_item = :item_id,
    n_epochs = 2,
    learning_rate = 0.01,
    n_negatives = 1,
    verbose = -1,
    kwargs...,
)
    model.user_history = Dict()
    for (userid, history) in
        zip(make_u2i_dataset(table, col_user = col_user, col_item = col_item)...)
        model.user_history[userid] = history
    end

    table, model.user2uidx, model.item2iidx, model.iidx2item =
        make_idmap(table, col_user = col_user, col_item = col_item)

    n_user = length(keys(model.user2uidx))

    unique_items = collect(keys(model.iidx2item))
    n_item = length(unique_items)

    model.user_embedding = rand(model.dim, n_user)
    model.item_embedding = rand(model.dim, n_item)

    if model.use_bias
        model.user_bias = rand(n_user)
        model.item_bias = rand(n_item)
        model.μ = rand()
    end

    # callback is any callbale with same interface
    callbacks = append!(Any[LogTrainLoss()], callbacks)
    for cb in callbacks
        if typeof(cb) <: AbstractCallback
            initialize!(cb, model, col_user = col_user, col_item = col_item)
        end
    end

    for epoch = 1:n_epochs
        train_loss = 0
        n_sample = 0
        for row in Tables.rows(table)
            uidx = row[col_user]

            # update positive
            iidx = row[col_item]

            pred = predict(model, uidx, iidx)
            train_loss = (model.loss(pred, 1) + train_loss * n_sample) / (n_sample + 1)
            n_sample += 1

            grad_value = grad(model.loss, pred, 1)
            sgd!(model, uidx, iidx, grad_value, learning_rate)

            # negative samples
            for _ = 1:n_negatives
                iidx = rand(unique_items)

                pred = predict(model, uidx, iidx)
                train_loss = (model.loss(pred, 0) + train_loss * n_sample) / (n_sample + 1)
                n_sample += 1

                grad_value = grad(model.loss, pred, 0)
                sgd!(model, uidx, iidx, grad_value, learning_rate)
            end
        end

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
    model::ImplicitMF,
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
