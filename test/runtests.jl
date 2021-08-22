using Recommender
using Test, Documenter


const tests = [
    "dataset/movielens.jl",
    "dataset/data_split.jl",
    "dataset/data_utils.jl",
    "core/item_knn.jl",
    "core/implicit_mf.jl",
    "metric.jl",
]

for t in tests
    @testset "Test $t" begin
        include(t)
    end
end