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
