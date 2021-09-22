function make_idmap(table; col_user = :userid, col_item = :itemid)
    table, user2uidx = reindex_id_column(table, col_user)
    table, item2iidx = reindex_id_column(table, col_item)
    iidx2item = Dict(iidx => itemid for (itemid, iidx) in item2iidx)
    return table, user2uidx, item2iidx, iidx2item
end

struct StopTrain <: Exception end

abstract type AbstractCallback end

function initialize!(
    cb::AbstractCallback,
    model::AbstractRecommender;
    col_user = :userid,
    col_item = :itemid,
)

    throw("initialize! not implemented.")
end

struct LogTrainLoss <: AbstractCallback end

initialize!(
    cb::LogTrainLoss,
    model::AbstractRecommender;
    col_user = :userid,
    col_item = :itemid,
) = nothing

function (cb::LogTrainLoss)(model::AbstractRecommender, train_loss, epoch, verbose = -1)
    if verbose >= 1 && (epoch % verbose == 0)
        @info "epoch=$epoch: train_loss=$train_loss"
    end
end

"""
    EvaluateValidData(valid_metric::MeanMetric, valid_table, early_stopping_rounds, name = "val_metric")

Callback to monitor the validation metrics during training, and raise `StopTrain` exception if early stopping is requred.

# Constructor arguments
- `valid_metric`: monotring metric. See [Evaluation metrics](@ref) for the available ones.
- `valid_table`: any `Tables.jl`-compatible object for validation dataset.
- `early_stopping_rounds`: If the validation metric does not improve more than this epoches, the early stopping is invoked. If set to be less than 1, no early stopping is applied.
- `name`: metrics name to show on logger.

# Example
Use in the matrix factorizaion training.
```julia
ndcg10 = MeanNDCG(10)
cb = EvaluateValidData(ndcg10, test_table, 1, "val_NDCG")

model = ImplicitMF(16, true, 0.01)
fit!(
    model,
    train_table,
    10,
    callbacks = [cb],
    col_item = :movieid,
    n_epochs = 20,
    n_negatives = 1,
    learning_rate = 0.01,
    verbose = 1,
)
```
"""
mutable struct EvaluateValidData <: AbstractCallback
    valid_metric::MeanMetric
    name::Union{AbstractString,Symbol}
    valid_table::Any
    early_stopping_rounds::Int
    best_epoch::Int
    best_val_metric::Float64
    topk::Int
    val_xs::Any
    val_ys::Any

    EvaluateValidData(
        valid_metric::MeanMetric,
        valid_table,
        early_stopping_rounds,
        name = "val_metric",
    ) = new(
        valid_metric,
        name,
        valid_table,
        early_stopping_rounds,
        1,
        0.0,
        -1,
        nothing,
        nothing,
    )
end

function initialize!(
    cb::EvaluateValidData,
    model::AbstractRecommender;
    col_user = :userid,
    col_item = :itemid,
)
    # TODO: loss as metric
    cb.topk = cb.valid_metric.base_metric.k
    cb.val_xs, cb.val_ys =
        make_u2i_dataset(cb.valid_table, col_user = col_user, col_item = col_item)
    preds = predict_u2i(model, cb.val_xs, cb.topk, drop_history = true)
    cb.best_val_metric = cb.valid_metric(preds, cb.val_ys)
end

function (cb::EvaluateValidData)(
    model::AbstractRecommender,
    train_loss,
    epoch,
    verbose = -1,
)
    preds = predict_u2i(model, cb.val_xs, cb.topk, drop_history = true)
    current_metric = cb.valid_metric(preds, cb.val_ys)

    if current_metric > cb.best_val_metric
        cb.best_epoch = epoch
        cb.best_val_metric = current_metric
    end

    if verbose >= 1 && (epoch % verbose == 0)
        @info "epoch=$epoch: $(cb.name)=$current_metric, best_$(cb.name)=$(cb.best_val_metric), best_epoch=$(cb.best_epoch)"
    end

    if cb.early_stopping_rounds >= 1 &&
       ((epoch - cb.best_epoch) >= cb.early_stopping_rounds)
        throw(StopTrain)
    end
end
