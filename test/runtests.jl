using Recommender
using Test

const tests = ["load_movielens1m.jl"]

for t in tests
    @testset "Test $t" begin
        include(t)
    end
end