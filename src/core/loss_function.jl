σ(x::Real) = 1.0 / (1.0 + exp(-x))

abstract type LossFunction end

struct Logloss <: LossFunction end

function (logloss::Logloss)(logit, label)
    prob = σ(logit)
    if label == 1
        return -log(prob)
    elseif label == 0
        return -log(1 - prob)
    else
        throw(ArgumentError("label must be 0 or 1."))
    end
end

function grad(loss::Logloss, logit, label)
    return σ(logit) - label
end

struct BPRLoss <: LossFunction end

function (loss::BPRLoss)(pred)
    return -log(σ(pred))
end

function grad(loss::BPRLoss, pred)
    return -1 + σ(pred)
end
