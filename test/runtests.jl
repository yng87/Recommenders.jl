using Recommender
using Test

# const tests = ["load_movielens1m.jl", "kNNRecommender.jl"]
const tests = ["load_movielens.jl"]

for t in tests
    @testset "Test $t" begin
        include(t)
    end
end