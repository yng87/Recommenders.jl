using Test, DataFrames, SparseArrays, MLJ, Tables
using Recommender: tfidf, compute_similarity, kNNRecommender, fit!, retrieve

# tfidf
inter = sparse([1, 1, 1, 2], [1, 2, 3, 3], [1., 3., 5., 2.])
idf = [3 / (3 + 1e-6), 3 / (1 + 1e-6)]
idf = log.(idf) .+ 1
expected =
    sparse([1, 1, 1, 2], [1, 2, 3, 3], [1 * idf[1], 3 * idf[1], 5 * idf[1], 2 * idf[2]])
evaluated = tfidf(inter)
@test evaluated == expected

# similarity
X = sparse([1, 1, 2, 2], [1, 2, 1, 2], [1., 2., 2., 1.])
expected = sparse([2, 1], [1, 2], [4 / (5 + 1e-6), 4 / (5 + 1e-6)])
evaluated = compute_similarity(X, 1, 0.0)
@test evaluated ≈ expected

# test MLJ integration
inter = sparse([1, 2, 2], [1, 1, 2], [1., 2., 1.])
model = kNNRecommender(k = 1)
fit!(model, inter)
expected_similarity = sparse(
    [2, 1],
    [1, 2],
    [
        1 * 2 / (1 * sqrt(5) + 1e-6),
        1 * 2 / (1 * sqrt(5) + 1e-6),
    ]
)
@test model.similarity ≈ expected_similarity

# retirieve
user_history = sparse([1, 0])
@test retrieve(model, user_history, 1, drop_history=true) == [2]
@test retrieve(model, user_history, 1, drop_history=false) == [2]
@test retrieve(model, user_history, 2, drop_history=true) == [2]
@test retrieve(model, user_history, 2, drop_history=false) == [2, 1]