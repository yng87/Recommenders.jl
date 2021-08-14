function leave_one_out_split(df::DataFrame; col_group = :userid, col_sort = :timestamp)
    df_out = combine(
        groupby(df, col_group),
        sdf -> sort(sdf, col_sort, rev = true),
        s -> (temp_rank = 1:nrow(s),),
    )
    df_out[!, :data_split] .= :train
    df_out[df_out.temp_rank.==1, :data_split] .= :test
    df_out[df_out.temp_rank.==2, :data_split] .= :valid
    df_out = df_out[!, Not(:temp_rank)]
    return df_out
end

function ratio_split(df::DataFrame; train_ratio = 0.7, valid_ratio = 0.1)
    n = nrow(df)
    n_train = floor(Int, n * train_ratio)
    n_valid = floor(Int, n * valid_ratio)

    if n - n_train - n_valid <= 0
        throw(ArgumentError("Invalid train/valid/test ratio."))
    end

    idx = randperm(n)
    df[!, :data_split] .= :train
    df[idx[(n_train+1):(n_train+n_valid)], :data_split] .= :valid
    df[idx[(n_train+n_valid+1):end], :data_split] .= :test

    return df
end
