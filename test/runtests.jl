using Recommenders
using Test, Documenter


const tests = [
    "dataset/movielens.jl",
    "dataset/data_split.jl",
    "dataset/data_utils.jl",
    "core/item_knn.jl",
    "core/loss_function.jl",
    "metric.jl",
    "model/item_knn.jl",
    "model/implicit_mf.jl",
    "model/bpr.jl",
    "model/slim.jl",
]

for t in tests
    @testset "Test $t" begin
        include(t)
    end
end