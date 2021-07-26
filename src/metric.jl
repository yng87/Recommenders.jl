struct RecallAtK
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
