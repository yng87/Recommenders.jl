using Recommenders
using Test, Documenter


const tests = [
    "dataset/movielens.jl",
    "dataset/data_split.jl",
    "dataset/data_utils.jl",
    "core/item_knn.jl",
    "core/loss_function.jl",
    "core/randomwalk_bipartite.jl",
    "metric.jl",
    "model/most_popular.jl",
    "model/item_knn.jl",
    "model/implicit_mf.jl",
    "model/bpr.jl",
    "model/slim.jl",
    "model/randomwalk.jl",
]

for t in tests
    @testset "Test $t" begin
        include(t)
    end
end
