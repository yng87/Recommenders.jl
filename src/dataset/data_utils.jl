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

function reindex_id_column!(
    df::DataFrame,
    col_id::Symbol;
    suffix::Union{Symbol,AbstractString} = :_index,
)
    id2index = Dict(id => index for (index, id) in enumerate(unique(df[!, col_id])))
    df[!, "$(col_id)$(suffix)"] = map(x -> id2index[x], df[!, col_id])
    return df, id2index
end

function make_u2i_dataset(table; col_user = :userid, col_item = :itemid)
    user_actioned_items = Dict()
    for row in eachrow(table)
        uid = row[col_user]
        iid = row[col_item]
        if uid in keys(user_actioned_items)
            push!(user_actioned_items[uid], iid)
        else
            user_actioned_items[uid] = [iid]
        end
    end
    xs = collect(keys(user_actioned_items))
    ys = collect(values(user_actioned_items))
    return xs, ys
end