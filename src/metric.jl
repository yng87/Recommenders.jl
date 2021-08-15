abstract type AbstractMetric end

struct RecallAtK <: AbstractMetric
    k::Int64
end

function (recall::RecallAtK)(recommends, ys)
    rec = 0
    for (pred, gt) in zip(recommends, ys)
        if pred === nothing || length(pred) == 0
            continue
        end
        k = min(recall.k, length(pred))
        pred = pred[1:k]
        num_hits = length(intersect(pred, gt))
        rec += num_hits / length(gt)
    end
    return rec / length(ys)
end

struct PrecisionAtK <: AbstractMetric
    k::Int64
end

function (precision::PrecisionAtK)(recommends, ys)
    rec = 0
    for (pred, gt) in zip(recommends, ys)
        if pred === nothing || length(pred) == 0
            continue
        end
        k = min(precision.k, length(pred))
        pred = pred[1:k]
        num_hits = length(intersect(pred, gt))
        rec += num_hits / length(pred)
    end
    return rec / length(ys)
end