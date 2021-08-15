using CSV, SparseArrays, DataFrames
using Recommender: rows2sparse, reindex_id_column, make_u2i_dataset
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
    table, id2index = reindex_id_column(df, :id)

    table = table |> Tables.columntable

    expected_table = (id = [1, 2, 3, 1],)
    expected_mapper = Dict(1 => 1, 4 => 2, 2 => 3)

    @test table == expected_table
    @test id2index == expected_mapper
end

@testset "Make u2i dataset" begin
    df = DataFrame(userid = [1, 1, 2, 3], itemid = [3, 4, 4, 1])
    xs, ys = make_u2i_dataset(df)
    sorted_idx = sortperm(xs)
    xs = xs[sorted_idx]
    ys = ys[sorted_idx]

    @test xs == [1, 2, 3]
    @test ys == [[3, 4], [4], [1]]
end
