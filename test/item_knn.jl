using Test, DataFrames, SparseArrays, MLJ, Tables
using Recommender: tfidf, compute_similarity, ItemkNN, fit, retrieve

# tfidf
inter = sparse([1, 1, 1, 2], [1, 2, 3, 3], [1.0, 3.0, 5.0, 2.0])
idf = [3 / (3 + 1e-6), 3 / (1 + 1e-6)]
idf = log.(idf) .+ 1
expected =
    sparse([1, 1, 1, 2], [1, 2, 3, 3], [1 * idf[1], 3 * idf[1], 5 * idf[1], 2 * idf[2]])
evaluated = tfidf(inter)
@test evaluated == expected

# similarity
X = sparse([1, 1, 2, 2], [1, 2, 1, 2], [1.0, 2.0, 2.0, 1.0])
expected = sparse([2, 1], [1, 2], [4 / (5 + 1e-6), 4 / (5 + 1e-6)])
evaluated = compute_similarity(X, 1, 0.0, true)
@test evaluated ≈ expected

# test MLJ integration
inter = (userid = [1, 2, 2], itemid = [1, 1, 2], rating = [1.0, 2.0, 1.0])
model = ItemkNN(k = 1)
mach = machine(model, inter)
fit!(mach)
expected_similarity =
    sparse([2, 1], [1, 2], [1 * 2 / (1 * sqrt(5) + 1e-6), 1 * 2 / (1 * sqrt(5) + 1e-6)])
@test mach.fitresult[1] ≈ expected_similarity

# retirieve
user_history = sparse([1, 0])
@test retrieve(model, mach.fitresult, user_history, 1) == [2]
@test retrieve(model, mach.fitresult, user_history, 2) == [2, 1]

@test retrieve(mach, user_history, 1) == [2]
@test retrieve(mach, user_history, 2) == [2, 1]

# @test retrieve(model, mach.fitresult, user_history, 1, drop_history = true) == [2]
# @test retrieve(model, mach.fitresult, user_history, 1, drop_history = false) == [2]
# @test retrieve(model, mach.fitresult, user_history, 2, drop_history = true) == [2]
# @test retrieve(model, mach.fitresult, user_history, 2, drop_history = false) == [2, 1]
