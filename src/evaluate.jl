"""
    hitrate(recommends, ys)

Compute mean HR.
# arguments
- recommends: recommends[i] is list of predictions for i-th input.
- ys: ys[i] is ground-truth for i-th input.

```@meta
DocTestSetup = quote
    using Recommender
end
```

```jldoctest
julia> hitrate([[1, 3], nothing, [2, 4], []], [1, 2, 3, 4])
0.25
```
"""
function hitrate(recommends, ys)
    hr = 0
    for (y, r) in zip(ys, recommends)
        if isnothing(r) continue end
        if y in r hr += 1 end
    end
    return hr / length(ys)
end