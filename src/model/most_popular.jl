mutable struct MostPopular <: AbstractRecommender
    df_popular::Union{Nothing,DataFrame}
    col_item::Union{Nothing,Symbol}
    MostPopular() = new(nothing, nothing)
end

function fit!(model::MostPopular, table; kwargs...)
    col_item = get(kwargs, :col_item, :itemid)
    df = DataFrame(table)
    model.col_item = col_item
    model.df_popular = sort(combine(groupby(df, col_item), nrow), [:nrow], rev = true)

    return model
end

function predict_u2i(model::MostPopular, userids, n::Int64; kwargs...)
    n = min(n, nrow(model.df_popular))
    pred = model.df_popular[!, model.col_item][1:n]
    return [pred for _ in eachindex(userids)]
end