@doc raw"""
    BPR(dim::Int64, reg_coeff::Float64)

Bayesian personalized ranking model. The model evaluates user-item triplet ``(u ,i ,j)``, which expresses "the user ``u`` prefers item ``i`` to item ``j``. Here the following matrix factoriazation model is adopted to model this relation:

```math
p_{uij} = \bm u_u \cdot \bm v_i - \bm u_u \cdot \bm v_j
```

# Constructor arguments
- `dim`: dimension of user/item vectors.
- `reg_coeff`: ``L_2`` regularization coefficients for model parameters.
"""
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


"""
    fit!(model::BPR, table; callbacks = Any[], col_user = :userid, col_item = :item_id, n_epochs = 2, learning_rate = 0.01, n_negatives = 1, verbose = -1)

Fit the `BPR` model by stochastic grandient descent. Instead the learnBPR algorithm proposed by the original paper, the simple SGD with negative sampling is implemented.

# Model-specific arguments
- `n_epochs`: number of epochs. During one epoch, all the row in `table` is read once.
- `learning_rate`: Learing rate of SGD.
- `n_negatives`: Number of negative item samples per positive (user, item) pair.
- `verbose`: If set to positive integer, the training info is printed once per `verbose`.
- `callbacks`: Additional callback functions during SGD. One can implement, for instance, monitoring the validation metrics and the early stopping. See [Callbacks](@ref).

# References
Rendel et. al. (2012), [BPR: Bayesian Personalized Ranking from Implicit Feedback](http://arxiv.org/abs/1205.2618)
"""
function fit!(
    model::BPR,
    table;
    callbacks = Any[],
    col_user = :userid,
    col_item = :item_id,
    n_epochs = 2,
    n_negatives = 1,
    learning_rate = 0.01,
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
            iidx = row[col_item]

            # sample negative
            for _ = 1:n_negatives
                jidx = rand(unique_items)
                while jidx in model.user_history[uidx]
                    jidx = rand(unique_items)
                end

                pred = predict(model, uidx, iidx, jidx)
                train_loss = (model.loss(pred) + train_loss * n_sample) / (n_sample + 1)
                n_sample += 1

                grad_value = grad(model.loss, pred)
                sgd!(model, uidx, iidx, jidx, grad_value, learning_rate)
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

@doc raw"""
    predict_u2i(model::BPR, userid::Union{AbstractString,Int}, n::Int64; drop_history = false)

Make predictions by using ``\bm u_u \cdot \bm v_i``.
"""
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
