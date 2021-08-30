function make_idmap(table; col_user = :userid, col_item = :itemid)
    table, user2uidx = reindex_id_column(table, col_user)
    table, item2iidx = reindex_id_column(table, col_item)
    iidx2item = Dict(iidx => itemid for (itemid, iidx) in item2iidx)
    return table, user2uidx, item2iidx, iidx2item
end


abstract type AbstractCallback end

mutable struct EvaluateValidData <: AbstractCallback
    valid_metric::MeanMetric
    valid_table::Any
    early_stopping_rounds::Int
    best_epoch::Int
    best_val_metric::Float64
    topk::Int
    val_xs::Any
    val_ys::Any
end

function EvaluateValidData(
    model::AbstractRecommender,
    valid_metric::MeanMetric,
    valid_table,
    early_stopping_rounds::Int;
    col_user = :userid,
    col_item = :itemid,
    drop_history = true,
)
    topk = valid_metric.base_metric.k
    val_xs, val_ys = make_u2i_dataset(valid_table, col_user = col_user, col_item = col_item)
    preds = predict_u2i(model, val_xs, topk, drop_history = drop_history)
    best_val_metric = valid_metric(preds, val_ys)
    # TODO: loss as metric

    return EvaluateValidData(
        valid_metric,
        valid_table,
        early_stopping_rounds,
        1,
        best_val_metric,
        topk,
        val_xs,
        val_ys,
    )
end

function call(
    cb::EvaluateValidData,
    model::AbstractRecommender,
    epoch;
    drop_history = true,
    rev = false,
)
    preds = predict_u2i(model, cb.val_xs, cb.topk, drop_history = drop_history)
    current_metric = cb.valid_metric(preds, cb.val_ys)

    sgn = ifelse(rev, -1, 1)
    if sgn * current_metric > sgn * cb.best_val_metric
        cb.best_epoch = epoch
        cb.best_val_metric = current_metric
    end

    stop_train = false
    if cb.early_stopping_rounds >= 1 &&
       ((epoch - cb.best_epoch) >= cb.early_stopping_rounds)
        stop_train = true
    end

    return current_metric, stop_train
end
