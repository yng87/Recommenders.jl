"""
    BaseRecommender

Abstract type for model objects.
"""
abstract type BaseRecommender end

function fit!(model::BaseRecommender, data)
    error("fit method is not implemented.")
end

function predict(model::BaseRecommender, u, i)
    error("predict method is not implemented.")
end

function retrieve(model::BaseRecommender, u, k)
    error("retrieve method is not implemented.")
end