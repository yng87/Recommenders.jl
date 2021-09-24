"""
    MostPopular()

Non-personalized baseline model which predicts n-most popular items in the corpus.
"""
mutable struct MostPopular <: AbstractRecommender
    df_popular::Union{Nothing,DataFrame}
    col_item::Union{Nothing,Symbol}
    user_histories::Any
    MostPopular() = new(nothing, nothing, nothing)
end

"""
    fit!(model::MostPopular, table; col_user = :userid, col_item = :itemid)

Fit most popular model.
"""
function fit!(model::MostPopular, table; col_user = :userid, col_item = :itemid, kwargs...)
    df = DataFrame(table)
    model.col_item = col_item
    model.df_popular = sort(combine(groupby(df, col_item), nrow), [:nrow], rev = true)

    users, items = make_u2i_dataset(table, col_user = col_user, col_item = col_item)
    model.user_histories = Dict()
    for (user, item) in zip(users, items)
        model.user_histories[user] = item
    end
    return model
end

"""
    predict_u2i(model::MostPopular, userid::Union{AbstractString,Int}, n::Int64; drop_history::Bool = false)

Make `n` prediction to user by most popular model.
"""
function predict_u2i(
    model::MostPopular,
    userid::Union{AbstractString,Int},
    n::Int64;
    drop_history::Bool = false,
    kwargs...,
)
    pred = model.df_popular[!, model.col_item]
    if drop_history && (userid in keys(model.user_histories))
        pred = filter(p -> !(p in model.user_histories[userid]), pred)
    end
    n = min(n, length(pred))
    return pred[1:n]
end
