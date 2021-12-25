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
    weighting_at_inference::Bool
    normalize::Bool
    normalize_similarity::Bool

    similarity::Any
    user_histories::Any
    user2uidx::Union{Dict,Nothing}
    item2iidx::Union{Dict,Nothing}
    iidx2item::Union{Dict,Nothing}

    ItemkNN(
        k::Int64,
        shrink::Float64,
        weighting::Union{Nothing,Symbol},
        weighting_at_inference::Bool,
        normalize::Bool,
        normalize_similarity::Bool,
    ) = new(
        k,
        shrink,
        weighting,
        weighting_at_inference,
        normalize,
        normalize_similarity,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
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

    @info "Make id map"
    table, model.user2uidx, model.item2iidx, model.iidx2item =
        make_idmap(table, col_user = col_user, col_item = col_item)

    @info "To sparse"
    X = rows2sparse(
        table,
        col_user = col_user,
        col_item = col_item,
        col_rating = col_rating,
    )

    if !model.weighting_at_inference
        model.user_histories = X
    end

    if model.weighting == :tfidf
        X = tfidf(X)
    elseif model.weighting == :bm25
        X = bm25(X)
    end

    if model.weighting_at_inference
        model.user_histories = X
    end

    @info "Compute similarity."
    model.similarity = compute_similarity(
        X,
        model.k,
        model.shrink,
        model.normalize,
        model.normalize_similarity,
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
            model.similarity,
            model.user_histories[uidx, :],
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
    pred = predict_i2i(model.similarity, iidx, n)
    return [model.iidx2item[iidx] for iidx in pred]
end
