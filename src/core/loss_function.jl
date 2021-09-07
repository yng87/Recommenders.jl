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

struct ElasticNet <: LossFunction
    α::Real
    l1_ratio::Real
end

function (loss::ElasticNet)(X, y, w)
    n_samples = length(y)
    α = loss.α
    l1_ratio = loss.l1_ratio
    loss = 1/(2*n_samples) * sum((y - X*w).^2) + α*l1_ratio*sum(abs.(w)) + 0.5*α*(1-l1_ratio)*sum(w.^2)
    return loss
end

function cd!(loss::ElasticNet, X, y, w)
    n = length(y)
    α = loss.α
    ρ = loss.l1_ratio
    γ = n*α*ρ

    for j in 1:n
        denom = sum(X[:, j].^2) + n*α*(1-ρ)

        z = y' * X[:, j] - (X[:, j]' * X * w - X[:, j]' * X[:, j] * w[j])
        signz = ifelse(z>=0, 1, -1)

        if (signz>0 && z > γ)
            w[j]=(z-γ)/denom
        elseif (signz<0 && z<-γ)
            w[j] = (z+γ)/denom
        else
            w[j]=0
        end
    end
end