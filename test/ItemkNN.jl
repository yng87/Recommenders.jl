using Test, DataFrames, SparseArrays
using Recommender: transform2sparse, tfidf, compute_similarity_matrix

X = DataFrame(:userid=>[10, 10, 10, 30], :itemid=>[400, 500, 600, 600], :target=>[1, 3, 5, 2])
user2uidx = Dict(10=>1, 30=>2)
item2iidx = Dict(400=>1, 500=>2, 600=>3)
Xsparse = transform2sparse(X, user2uidx, item2iidx)

expected = sparse([1, 1, 1, 2], [1, 2, 3, 3], [1., 3., 5., 2.])
@test Xsparse == expected

idf = [3/(3 + 1e-6), 3/(1 + 1e-6)]
idf = log.(idf) .+ 1
expected = sparse([1, 1, 1, 2], [1, 2, 3, 3], [1*idf[1], 3*idf[1], 5*idf[1], 2*idf[2]])
evaluated = tfidf(Xsparse)
@test evaluated == expected

X = sparse([1, 1, 2, 2], [1, 2, 1, 2], [1., 2., 2., 1.])
expected = sparse([2, 1], [1, 2], [4/(5+1e-6), 4/(5+1e-6)])
evaluated = compute_similarity_matrix(X, 1, 0.)
@test evaluated ≈ expected

# TODO: write MLJ test