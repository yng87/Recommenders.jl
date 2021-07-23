struct LeaveOneOut <: MLJ.ResamplingStrategy
    time_column::Symbol
    key_column::Symbol
    rng::Union{Int,AbstractRNG}

    function LeaveOneOut(time_column, key_column, rng)
        return new(time_column, key_column, rng)
    end
end

# Keyword Constructor
function LeaveOneOut(;
    time_column::Symbol = :timestamp,
    key_column::Symbol = :userid,
    rng = nothing,
)
    if rng isa Integer
        rng = MersenneTwister(rng)
    end
    if rng === nothing
        rng = Random.GLOBAL_RNG
    end

    return LeaveOneOut(time_column, key_column, rng)
end

function train_test_pairs(loo::LeaveOneOut, rows, X)
    if !(X isa DataFrame)
        X = DataFrame(X)
    end
    df = X[rows, [loo.key_column, loo.time_column]]
    df[!, :original_rows] = rows

    df[!, :train_val_test_split] .= :train
    df = combine(
        groupby(df, loo.key_column),
        sdf -> sort(sdf, loo.time_column, rev = true),
        s -> (rank = 1:nrow(s),),
    )
    df[df.rank.==1, :train_val_test_split] .= :test
    sort!(df, :original_rows)

    train = rows[df.train_val_test_split.==:train]
    test = rows[df.train_val_test_split.==:test]
    return [(train, test)]
end
