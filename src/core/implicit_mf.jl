function σ(x::Real)
    # sigmoid
    if x > 0
        return 1.0 / (1.0 + exp(-x))
    else
        return exp(x) / (exp(x) + 1)
    end
end

function logloss(pred, label)
    # pred is logit
    prob = σ(pred)
    if label == 1
        return -log(prob)
    elseif label == 0
        return -log(1 - prob)
    else
        throw(ArgumentError("label must be 0 or 1."))
    end
end

function grad_logloss(pred, label)
    return σ(pred) - label
end