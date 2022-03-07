@doc raw"""
    ItemkNN(k::Int64, shrink::Float64, weighting::Union{Nothing,Symbol}, weighting_at_inference::Bool, normalize::Bool, normalize_similarity::Bool)

Item-based k-nearest neighborhood algorithm with cosine similarity.
The model first computes the item-to-item similarity matrix

```math
s_{ij} = \frac{\bm r_i \cdot \bm r_j}{\|\bm r_i\|\|\bm r_j\| + h}\,,
```
where ``r_{i,u}`` is rating for item ``i`` by user ``u`` and ``h`` is the shrink parameter to suppress the contributions from items with a few ratings.

# Constructor arguments
- `k`: size of the nearest neighbors. Only the k-most similar items to each item are stored, which reduces sparse similarity maxrix size, and also make better predictions.
- `shrink`: shrink paramerer explained above.
- `weighting`: if set to `:ifidf` or `:bm25`, the raw rating matrix is weighted by TF-IDF or BM25, respectively, before computing similarity. If not necessary, just set `nothing`.
- `weighting_at_inference`: to use above weighting at inference time, only relevant for BM25.
- `normalize_similarity`: if set to `true`, normalize each column of similarity matrix. See the reference for detail.

# References
M. Deshpande and G. Karypis (2004), [Item-based top-N recommendation algorithms](https://doi.org/10.1145/963770.963776).
"""
mutable struct ItemkNN <: AbstractRecommender
    k::Int64
    shrink::Float64
    weighting::Union{Nothing,Symbol}
    normalize::Bool
    normalize_similarity::Bool
    include_self::Bool

    similar_items::Vector{Int}
    similarity_scores::Vector{Float64}
    user_histories::Dict{Int,Vector{Int}}
    user2uidx::Dict{Union{Int,AbstractString},Int}
    item2iidx::Dict{Union{Int,AbstractString},Int}
    iidx2item::Dict{Int,Union{Int,AbstractString}}

    ItemkNN(
        k::Int64,
        shrink::Float64,
        weighting::Union{Nothing,Symbol},
        normalize::Bool,
        normalize_similarity::Bool,
        include_self::Bool = true,
    ) = new(
        k,
        shrink,
        weighting,
        normalize,
        normalize_similarity,
        include_self,
        Int[],
        Float64[],
        Dict(),
        Dict(),
        Dict(),
        Dict(),
    )
end

"""
    fit!(model::ItemkNN, table; col_user = :userid, col_item = :itemid, col_rating = :rating)

Fit the `ItemkNN` model. `col_rating` specifies rating column in the `table`, which will be all unity if implicit feedback data is given.
"""
function fit!(model::ItemkNN, table; kwargs...)
    col_user = get(kwargs, :col_user, :userid)
    col_item = get(kwargs, :col_item, :itemid)
    col_rating = get(kwargs, :col_rating, :rating)

    @debug "Build lookup."
    table, model.user2uidx, model.item2iidx, model.iidx2item =
        make_idmap(table, col_user = col_user, col_item = col_item)

    table = Tables.dictcolumntable(table)

    if model.weighting == :tfidf
        @debug "Calculate TF-IDF."
        table =
            tfidf(table, col_user = col_user, col_item = col_item, col_rating = col_rating)
    elseif model.weighting == :bm25
        @debug "Calculate BM25."
        table =
            bm25(table, col_user = col_user, col_item = col_item, col_rating = col_rating)
    end

    @debug "Prepare sparse rating hisotry."
    uidx2rated_itmes, iidx2rated_users, uidx2rating, iidx2rating = get_rating_history(
        table,
        col_user = col_user,
        col_item = col_item,
        col_rating = col_rating,
    )

    @debug "Cache user history."
    model.user_histories = Dict{Int,SparseVector}()
    n_items = length(keys(model.iidx2item))
    for uidx in keys(uidx2rated_itmes)
        rated_items = uidx2rated_itmes[uidx]
        model.user_histories[uidx] = rated_items
    end

    @debug "Calculate similarity."
    model.similar_items, model.similarity_scores = compute_similarity(
        uidx2rated_itmes,
        iidx2rated_users,
        uidx2rating,
        iidx2rating,
        model.k,
        model.shrink,
        model.normalize,
        model.normalize_similarity,
        model.include_self,
    )
    return model
end

@doc raw"""
    predict_u2i(model::ItemkNN, userid::Union{AbstractString,Int}, n::Int64; drop_history = false)

Recommend top-`n` items for user by `ItemkNN`. The predicted rating of item ``i`` by user ``u`` is computed by

```math

\hat{r}_{i, u} = \sum_{j} s_{ij} r_{j, u}\,,
```
where ``r_{j, u}`` is the actual user rating while ``\hat{r}_{i, u}`` is the model prediction.
"""
function predict_u2i(
    model::ItemkNN,
    userid::Union{AbstractString,Int},
    n::Int64;
    drop_history = false,
    kwargs...,
)
    if userid in keys(model.user2uidx)
        uidx = model.user2uidx[userid]
        pred = predict_u2i(
            model.similar_items,
            model.similarity_scores,
            model.user_histories[uidx],
            model.k,
            n,
            drop_history = drop_history,
        )
        return [model.iidx2item[iidx] for iidx in pred]

    else
        return []
    end
end

"""
    predict_i2i(model::ItemkNN, itemid::Union{AbstractString,Int}, n::Int64)

Make `n` prediction for a give item by ItenkNN model.
"""
function predict_i2i(model::ItemkNN, itemid::Union{AbstractString,Int}, n::Int64)
    if !(itemid in keys(model.item2iidx))
        return []
    end

    iidx = model.item2iidx[itemid]
    pred = predict_i2i(model.similar_items, iidx, model.k, n)
    return [model.iidx2item[iidx] for iidx in pred]
end
