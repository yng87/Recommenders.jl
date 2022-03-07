@doc raw"""
    ImplicitMF(dim::Int64, use_bias::Bool, reg_coeff::Float64)

Matrix factorization model for implicit feedback. The predicted rating for item ``i`` by user ``u`` is expreseed as
```math
\hat r_{ui} = \mu + b_i + b_u + \bm u_u \cdot \bm v_i\,,
```
Unlike the model for explicit feedback, the model treats all the (user, item) pairs in the train dataset as positive interaction with label 1, and sample negative (user, item) pairs from the corpus. Currently only the uniform item sampling is implemented. The fitting criteria is the ordinary logloss function
```math
    L = -r_{ui}\log(\hat r_{ui}) - (1 - r_{ui})\log(1 - \hat r_{ui}).
```

# Constructor arguments
- `dim`: dimension of user/item vectors.
- `use_bias`: if set to false, the bias terms (``\mu``, ``b_i``, ``b_u``) are set to zero.
- `reg_coeff`: ``L_2`` regularization coefficients for model parameters.

# References
For instance, Rendle et. al. (2020), [Neural Collaborative Filtering vs. Matrix Factorization Revisited
](http://arxiv.org/abs/2005.09683).
"""
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
    pred = view(model.user_embedding, :, uidx)' * view(model.item_embedding, :, iidx)
    if model.use_bias
        pred += model.μ + view(model.user_bias, uidx) + view(model.item_bias, iidx)
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


"""
    fit!(model::ImplicitMF, table; callbacks = Any[], col_user = :userid, col_item = :item_id, n_epochs = 2, learning_rate = 0.01, n_negatives = 1, verbose = -1)

Fit the `ImplicitMF` model by stochastic grandient descent (with no batching).

# Model-specific arguments
- `n_epochs`: number of epochs. During one epoch, all the row in `table` is read once.
- `learning_rate`: Learing rate of SGD.
- `n_negatives`: Number of negative item samples per positive (user, item) pair.
- `verbose`: If set to positive integer, the training info is printed once per `verbose`.
- `callbacks`: Additional callback functions during SGD. One can implement, for instance, monitoring the validation metrics and the early stopping. See [Callbacks](@ref).
"""
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

    n_sample = nothing
    for epoch = 1:n_epochs
        if epoch == 1
            p = ProgressUnknown("Epoch $(epoch): training...")
        else
            p = Progress(n_sample, 1, "Epoch $(epoch): training...")
        end
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
            next!(p)

            # negative samples
            for _ = 1:n_negatives
                iidx = rand(unique_items)

                pred = predict(model, uidx, iidx)
                train_loss = (model.loss(pred, 0) + train_loss * n_sample) / (n_sample + 1)
                n_sample += 1

                grad_value = grad(model.loss, pred, 0)
                sgd!(model, uidx, iidx, grad_value, learning_rate)
                next!(p)
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
    predict_u2i(model::ImplicitMF, userid::Union{AbstractString,Int}, n::Int64; drop_history = false)

Make predictions by using ``\hat r_{ui}``.
"""
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
