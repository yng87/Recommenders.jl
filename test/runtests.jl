using Recommender
using Test

const tests = ["load_movielens.jl", "item_knn.jl"]

for t in tests
    @testset "Test $t" begin
        include(t)
    end
end