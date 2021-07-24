function rows2sparse(X; col_user = :userid, col_item = :itemid, col_rating = :rating)
    U = Int[]
    I = Int[]
    R = Float64[]

    for row in Tables.rows(X)
        push!(U, row[col_user])
        push!(I, row[col_item])
        push!(R, row[col_rating])
    end

    return sparse(U, I, R)
end

function reindex_id_column!(df::DataFrame, col_id::Symbol; suffix::Symbol = :index)
    id2index = Dict(id => index for (index, id) in enumerate(unique(df[!, col_id])))
    df[!, "$(col_id)_$(suffix)"] = map(x -> id2index[x], df[!, col_id])
    return df, id2index
end
