using Test, SparseArrays, Tables
using Recommenders: tfidf, bm25, get_rating_history, compute_similarity, predict_u2i

@testset "TF-IDF" begin
    inter = Tables.dictcolumntable(
        Dict(
            :userid => [1, 1, 1, 2],
            :itemid => [1, 2, 3, 3],
            :rating => [1.0, 3.0, 5.0, 2.0],
        ),
    )
    idf = [3 / (3 + 1), 3 / (1 + 1)]
    idf = log.(idf)
    expected = Tables.dictcolumntable(
        Dict(
            :userid => [1, 1, 1, 2],
            :itemid => [1, 2, 3, 3],
            :rating => [1 * idf[1], 3 * idf[1], 5 * idf[1], 2 * idf[2]],
        ),
    )
    evaluated = tfidf(inter)
    @test evaluated[:userid] == expected[:userid]
    @test evaluated[:itemid] == expected[:itemid]
    @test evaluated[:rating] == expected[:rating]
end

@testset "BM25" begin
    k1 = 1.2
    b = 0.75
    inter = Tables.dictcolumntable(
        Dict(
            :userid => [1, 1, 1, 2],
            :itemid => [1, 2, 3, 3],
            :rating => [1.0, 3.0, 5.0, 2.0],
        ),
    )
    idf = [(3 - 3 + 0.5) / (3 + 0.5), (3 - 1 + 0.5) / (1 + 0.5)]
    idf = log.(idf)
    dl = [1, 1, 2]
    avgdl = 4 / 3
    dl_factor = [
        (k1 + 1) / (1 + k1 * (1 - b + b * dl[1] / avgdl)),
        (k1 + 1) / (1 + k1 * (1 - b + b * dl[2] / avgdl)),
        (k1 + 1) / (1 + k1 * (1 - b + b * dl[3] / avgdl)),
    ]
    expected = Tables.dictcolumntable(
        Dict(
            :userid => [1, 1, 1, 2],
            :itemid => [1, 2, 3, 3],
            :rating => [
                1 * idf[1] * dl_factor[1],
                3 * idf[1] * dl_factor[2],
                5 * idf[1] * dl_factor[3],
                2 * idf[2] * dl_factor[3],
            ],
        ),
    )
    evaluated = bm25(inter, k1, b)
    @test evaluated[:userid] ≈ expected[:userid]
    @test evaluated[:itemid] ≈ expected[:itemid]
    @test evaluated[:rating] ≈ expected[:rating]
end

@testset "Compute similarity matrix" begin
    X = Tables.dictcolumntable(
        Dict(
            :userid => [1, 1, 2, 2],
            :itemid => [1, 2, 1, 2],
            :rating => [1.0, 2.0, 2.0, 1.0],
        ),
    )
    expected = sparse([2, 1], [1, 2], [4 / (5 + 1e-6), 4 / (5 + 1e-6)])
    uidx2rated_itmes, iidx2rated_users, uidx2rating, iidx2rating = get_rating_history(X)

    evaluated = compute_similarity(
        uidx2rated_itmes,
        iidx2rated_users,
        uidx2rating,
        iidx2rating,
        1,
        0.0,
        true,
        false,
        false,
    )
    @test evaluated ≈ expected

    evaluated = compute_similarity(
        uidx2rated_itmes,
        iidx2rated_users,
        uidx2rating,
        iidx2rating,
        1,
        0.0,
        true,
        true,
        false,
    )
    for c in eachcol(evaluated)
        @test sum(c) ≈ 1
    end

    evaluated = compute_similarity(
        uidx2rated_itmes,
        iidx2rated_users,
        uidx2rating,
        iidx2rating,
        1,
        0.0,
        true,
        false,
        true,
    )
    @test evaluated[1, 1] > 0
    @test evaluated[2, 2] > 0
end

@testset "Predict user to item" begin
    inter = (userid = [1, 2, 2], itemid = [1, 1, 2], rating = [1.0, 2.0, 1.0])
    similarity = sparse([2, 1], [1, 2], [4 / (5 + 1e-6), 4 / (5 + 1e-6)])
    user_history = sparse([1, 0])
    @test predict_u2i(similarity, user_history, 1) == [2]
    @test predict_u2i(similarity, user_history, 2) == [2] # score=0 なので一個しか返らない
    @test predict_u2i(similarity, user_history, 1, drop_history = true) == [2]
    @test predict_u2i(similarity, user_history, 2, drop_history = true) == [2]
    # check dispatch
    @test predict_u2i(similarity, [1, 0], 1) == [2]
end

@testset "Predict item to item" begin
    similarity = sparse([2, 1], [1, 2], [4 / (5 + 1e-6), 4 / (5 + 1e-6)])
    @test predict_i2i(similarity, 1, 1) == [2]
    @test predict_i2i(similarity, 2, 1) == [1]
    @test predict_i2i(similarity, 1, 2) == [2]
    @test predict_i2i(similarity, 2, 2) == [1]
end
