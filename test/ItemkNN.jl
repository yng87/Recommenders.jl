using Test, DataFrames, SparseArrays, MLJ, Tables
using Recommender: transform2sparse, tfidf, compute_similarity_matrix, ItemkNN, predict_i2i, predict_u2i

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

# test MLJ integration
X = DataFrame(:userid=>[10, 10, 30, 30], :itemid=>[400, 500, 400, 500], :target=>[1, 2, 2, 1])
model = ItemkNN(k=1)
knn = machine(model, X)
fit!(knn)
expected_fitresult = (
    expected,
    Dict(10=>1, 30=>2),
    Dict(400=>1, 500=>2)
)
@test knn.fitresult[1] ≈ expected_fitresult[1]
@test knn.fitresult[2] == expected_fitresult[2]
@test knn.fitresult[3] == expected_fitresult[3]
@test knn.cache === nothing
@test knn.report === nothing

Xnew = DataFrame(:userid=>[10, 30, 10, 50], :itemid=>[400, 500, 600, 400], :target=>[1, 2, 2, 1])
expected_preds = DataFrame(:userid=>[10, 30, 50], :preds=>[[500], [400], [500]])
@test predict_u2i(knn, Xnew) == expected_preds

# test MLJ by Tables.columntable format
X = X |>  Tables.columntable
model = ItemkNN(k=1)
knn = machine(model, X)
fit!(knn)
@test knn.fitresult[1] ≈ expected_fitresult[1]
@test knn.fitresult[2] == expected_fitresult[2]
@test knn.fitresult[3] == expected_fitresult[3]
@test knn.cache === nothing
@test knn.report === nothing

Xnew = Xnew |> Tables.columntable
@test predict_u2i(knn, Xnew) == expected_preds