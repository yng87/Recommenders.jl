using Test, SparseArrays
using Recommender: tfidf, bm25, compute_similarity, predict_u2i

@testset "TF-IDF" begin
    inter = sparse([1, 1, 1, 2], [1, 2, 3, 3], [1.0, 3.0, 5.0, 2.0])
    idf = [3 / (3 + 1), 3 / (1 + 1)]
    idf = log.(idf)
    expected =
        sparse([1, 1, 1, 2], [1, 2, 3, 3], [1 * idf[1], 3 * idf[1], 5 * idf[1], 2 * idf[2]])
    evaluated = tfidf(inter)
    @test evaluated == expected
end

@testset "BM25" begin
    k1 = 1.2
    b = 0.75
    inter = sparse([1, 1, 1, 2], [1, 2, 3, 3], [1.0, 3.0, 5.0, 2.0])
    idf = [(3 - 3 + 0.5) / (3 + 0.5), (3 - 1 + 0.5) / (1 + 0.5)]
    idf = log.(idf)
    dl = [1, 1, 2]
    avgdl = 4 / 3
    dl_factor = [
        (k1 + 1) / (1 + k1 * (1 - b + b * dl[1] / avgdl)),
        (k1 + 1) / (1 + k1 * (1 - b + b * dl[2] / avgdl)),
        (k1 + 1) / (1 + k1 * (1 - b + b * dl[3] / avgdl)),
    ]
    expected = sparse(
        [1, 1, 1, 2],
        [1, 2, 3, 3],
        [
            1 * idf[1] * dl_factor[1],
            3 * idf[1] * dl_factor[2],
            5 * idf[1] * dl_factor[3],
            2 * idf[2] * dl_factor[3],
        ],
    )
    evaluated = bm25(inter, k1, b)
    @test evaluated ≈ expected
end

@testset "Compute similarity matrix" begin
    X = sparse([1, 1, 2, 2], [1, 2, 1, 2], [1.0, 2.0, 2.0, 1.0])
    expected = sparse([2, 1], [1, 2], [4 / (5 + 1e-6), 4 / (5 + 1e-6)])
    evaluated = compute_similarity(X, 1, 0.0, true)
    @test evaluated ≈ expected
end

@testset "Predict user to item" begin
    inter = (userid = [1, 2, 2], itemid = [1, 1, 2], rating = [1.0, 2.0, 1.0])
    similarity = sparse([2, 1], [1, 2], [4 / (5 + 1e-6), 4 / (5 + 1e-6)])
    user_history = sparse([1, 0])
    @test predict_u2i(similarity, user_history, 1) == [2]
    @test predict_u2i(similarity, user_history, 2) == [2, 1]
end