using DataFrames
using Recommender: leave_one_out_split, ratio_split
using Test

@testset "Leave one out split by timestamp" begin
    df = DataFrame((
        userid = [1, 1, 1, 2, 2, 3, 3, 3, 3],
        timestamp = [0, 2, 10, 3, 5, 4, 5, 11, -1],
    ))
    df_processed = leave_one_out_split(df, col_group = :userid, col_sort = :timestamp)

    expected_df = DataFrame((
        userid = [1, 1, 1, 2, 2, 3, 3, 3, 3],
        timestamp = [0, 2, 10, 3, 5, 4, 5, 11, -1],
        data_split = [:train, :valid, :test, :valid, :test, :train, :valid, :test, :train],
    ))

    @test sort(df_processed, [:userid, :timestamp]) ==
          sort(expected_df, [:userid, :timestamp])
end


@testset "Ratio split" begin
    df = DataFrame((c = [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],))
    df_processed = ratio_split(df, train_ratio = 0.5, valid_ratio = 0.3)
    @test sum(df.data_split .== :train) == 5
    @test sum(df.data_split .== :valid) == 3
    @test sum(df.data_split .== :test) == 2
end
