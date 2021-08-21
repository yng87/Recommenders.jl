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
    prec = 0
    for (pred, gt) in zip(recommends, ys)
        if pred === nothing || length(pred) == 0
            continue
        end
        k = min(precision.k, length(pred))
        pred = pred[1:k]
        num_hits = length(intersect(pred, gt))
        prec += num_hits / length(pred)
    end
    return prec / length(ys)
end


struct NDCG <: AbstractMetric
    k::Int64
end

function (precision::NDCG)(recommends, ys)
    ndcg = 0
    for (pred, gt) in zip(recommends, ys)
        if pred === nothing || length(pred) == 0
            continue
        end
        k = min(precision.k, length(pred))
        pred = pred[1:k]
        
        dcg = 0
        for i in eachindex(pred)
            if pred[i] in gt
                dcg += 1.0 / log2(1.0 + i)
            end
        end

        idcg=0
        for i in 1:min(length(pred), length(gt))
            idcg+= 1.0 / log2(1.0 + i)
        end

        ndcg += dcg / idcg
    end
    return ndcg / length(ys)
end