using CSV, SparseArrays, DataFrames
using Recommender: rows2sparse, reindex_id_column!
using Test

@testset "CSV to sparse matrix" begin
    csv_str = """
    col1,col2,col3
    1,1,0.1
    1,2,-1
    2,2,5
    """
    csv = CSV.File(IOBuffer(csv_str))
    mat_sparse = rows2sparse(csv, col_user = :col1, col_item = :col2, col_rating = :col3)
    expected = sparse([1, 1, 2], [1, 2, 2], [0.1, -1, 5])
    @test mat_sparse == expected
end

@testset "DataFrame to sparse matrix" begin
    df = DataFrame(col1 = [1, 1, 2], col2 = [1, 2, 2], col3 = [0.1, -1, 5])
    mat_sparse = rows2sparse(df, col_user = :col1, col_item = :col2, col_rating = :col3)
    expected = sparse([1, 1, 2], [1, 2, 2], [0.1, -1, 5])
    @test mat_sparse == expected
end

@testset "Re-index id column" begin
    df = DataFrame(id = [1, 4, 2, 1])
    df, id2index = reindex_id_column!(df, :id, suffix = :_reind)

    expected_df = DataFrame(id = [1, 4, 2, 1], id_reind = [1, 2, 3, 1])
    expected_mapper = Dict(1 => 1, 4 => 2, 2 => 3)

    @test df == expected_df
    @test id2index == expected_mapper
end
