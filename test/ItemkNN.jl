using Test, DataFrames, SparseArrays, MLJ, Tables
using Recommender: transform2sparse, tfidf, compute_similarity, ItemkNN, retrieve

X = DataFrame(:userid=>[10, 10, 10, 30], :itemid=>[400, 500, 600, 600], :target=>[1, 3, 5, 2])
Xsparse, user2uidx, item2iidx = transform2sparse(X)

expected = sparse([1, 1, 1, 2], [1, 2, 3, 3], [1., 3., 5., 2.])
@test Xsparse == expected
@test user2uidx == Dict(10=>1, 30=>2)
@test item2iidx == Dict(400=>1, 500=>2, 600=>3)


idf = [3/(3 + 1e-6), 3/(1 + 1e-6)]
idf = log.(idf) .+ 1
expected = sparse([1, 1, 1, 2], [1, 2, 3, 3], [1*idf[1], 3*idf[1], 5*idf[1], 2*idf[2]])
evaluated = tfidf(Xsparse)
@test evaluated == expected

X = sparse([1, 1, 2, 2], [1, 2, 1, 2], [1., 2., 2., 1.])
expected = sparse([2, 1], [1, 2], [4/(5+1e-6), 4/(5+1e-6)])
evaluated = compute_similarity(X, 1, 0.)
@test evaluated ≈ expected

# test MLJ integration
X = DataFrame(:userid=>[10, 30, 30], :itemid=>[400, 400, 500], :target=>[1, 2, 1])
model = ItemkNN(k=1)
knn = machine(model, X)
fit!(knn)
expected_fitresult = (
    sparse([2, 1], [1, 2],
            [1*2*(log(2/(2+1e-6)) + 1)^2 / (1 * sqrt(5) + 1e-6),
            1*2*(log(2/(2+1e-6)) + 1)^2 / (1 * sqrt(5) + 1e-6)]),
    sparse([1, 2, 2], [1, 1, 2], [1, 2, 1]),
    Dict(10=>1, 30=>2),
    Dict(400=>1, 500=>2),
    Dict(1=>400, 2=>500)
)
@test knn.fitresult[1] ≈ expected_fitresult[1] atol=1e-3
@test knn.fitresult[2] == expected_fitresult[2]
@test knn.fitresult[3] == expected_fitresult[3]
@test knn.fitresult[4] == expected_fitresult[4]
@test knn.fitresult[5] == expected_fitresult[5]
@test knn.cache === nothing
@test knn.report === nothing

Xnew = [10, 30, 50]
expected_preds = [[10, [500]], [30, nothing], [50, nothing]]
@test retrieve(knn, Xnew, 1) == expected_preds

# test MLJ by Tables.columntable format
X = X |>  Tables.columntable
model = ItemkNN(k=1)
knn = machine(model, X)
fit!(knn)
@test knn.fitresult[1] ≈ expected_fitresult[1] atol=1e-3
@test knn.fitresult[2] == expected_fitresult[2]
@test knn.fitresult[3] == expected_fitresult[3]
@test knn.fitresult[4] == expected_fitresult[4]
@test knn.fitresult[5] == expected_fitresult[5]
@test knn.cache === nothing
@test knn.report === nothing

@test retrieve(knn, Xnew, 1) == expected_preds