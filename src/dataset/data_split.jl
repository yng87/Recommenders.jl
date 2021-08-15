function leave_one_out_split(table; col_user = :userid, col_time = :timestamp)
    last_action = Dict()
    for row in Tables.rows(table)
        user = row[col_user]
        timestamp = row[col_time]
        if !(user in keys(last_action))
            last_action[user] = timestamp
        else
            last_action[user] = max(timestamp, last_action[user])
        end
    end

    train_table =
        table |> TableOperations.filter(row -> row[col_time] < last_action[row[col_user]])
    test_table =
        table |> TableOperations.filter(row -> row[col_time] == last_action[row[col_user]])

    m = Tables.materializer(table)

    return m(train_table), m(test_table)
end

function ratio_split(table, train_ratio = 0.7)
    (train_ratio < 0 || train_ratio > 1) &&
        throw(ArgumentError("train_ratio must be between 0 and 1."))
    # currently it requires table that is converted to DataFrame
    df = DataFrame(table)
    n = nrow(df)
    n_train = round(Int, n * train_ratio)
    idx = randperm(n)
    idx_train = idx[1:n_train]
    idx_test = idx[(n_train+1):end]
    df_train = df[idx_train, :]
    df_test = df[idx_test, :]

    m = Tables.materializer(table)

    return m(df_train), m(df_test)
end
