"""
    hitrate(recommends, ys)

```jldoctest
julia> hitrate([[1, 3], nothing, [2, 4]], [1, 2, 3])
0.3333333333333
````
"""
function hitrate(recommends, ys)
    hr = 0
    for (y, r) in zip(ys, recommends)
        if isnothing(r) continue end
        if y in r hr += 1 end
    end
    return hr / length(ys)
end