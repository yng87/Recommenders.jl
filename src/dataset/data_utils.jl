function rows2sparse(table; col_user = :userid, col_item = :itemid, col_rating = :rating)
    U = Int[]
    I = Int[]
    R = Float64[]

    for row in Tables.rows(table)
        push!(U, row[col_user])
        push!(I, row[col_item])
        push!(R, row[col_rating])
    end

    return sparse(U, I, R)
end

function reindex_id_column(table, col_id::Symbol)
    col = table |> TableOperations.select(col_id) |> Tables.columntable |> collect
    col = col[1]
    id2index = Dict(id => index for (index, id) in enumerate(unique(col)))
    reindexed_table = table |> TableOperations.transform(Dict(col_id => x -> id2index[x]))
    return Tables.materializer(table)(reindexed_table), id2index
end

function make_u2i_dataset(table; col_user = :userid, col_item = :itemid)
    user_actioned_items =
        Dict{Union{AbstractString,Int},Vector{Union{AbstractString,Int}}}()
    for row in Tables.rows(table)
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
