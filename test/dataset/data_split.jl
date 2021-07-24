using DataFrames
using Recommender: leave_one_out_split
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


