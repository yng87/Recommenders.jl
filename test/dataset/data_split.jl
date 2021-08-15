using DataFrames
using Recommender: leave_one_out_split, ratio_split
using Test

@testset "Leave one out split by timestamp" begin
    df = DataFrame(
        userid = [1, 1, 1, 2, 2, 3, 3, 3, 3],
        timestamp = [0, 2, 10, 3, 5, 4, 5, 11, -1],
    )
    df_train, df_test = leave_one_out_split(df, col_user = :userid, col_time = :timestamp)
    sort!(df_train, [:userid, :timestamp])
    sort!(df_test, [:userid, :timestamp])

    @test df_train ==
          DataFrame(userid = [1, 1, 2, 3, 3, 3], timestamp = [0, 2, 3, -1, 4, 5])
    @test df_test == DataFrame(userid = [1, 2, 3], timestamp = [10, 5, 11])
end


@testset "Ratio split" begin
    df = DataFrame((c = [1, 1, 1, 2, 2, 3, 3, 3, 3, 4],))
    df_train, df_test = ratio_split(df, 0.8)
    @test nrow(df_train) == 8
    @test nrow(df_test) == 2
end
