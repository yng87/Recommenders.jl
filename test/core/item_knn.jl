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
    # expected = sparse([2, 1], [1, 2], [4 / (5 + 1e-6), 4 / (5 + 1e-6)])
    expected_similar_items = [2, 1]
    expected_similarity = [4 / (5 + 1e-6), 4 / (5 + 1e-6)]
    uidx2rated_itmes, iidx2rated_users, uidx2rating, iidx2rating = get_rating_history(X)

    evaluated_similar_items, evaluated_similarity = compute_similarity(
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
    @test evaluated_similar_items == expected_similar_items
    @test evaluated_similarity ≈ expected_similarity

    evaluated_similar_items, evaluated_similarity = compute_similarity(
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
    @test evaluated_similarity ≈ [1.0, 1.0]

    expected_similar_items = [1, 2]
    evaluated_similar_items, evaluated_similarity = compute_similarity(
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
    @test evaluated_similar_items == expected_similar_items
end

@testset "Predict user to item" begin
    similar_items = [2, 1]
    similarity = [4 / (5 + 1e-6), 4 / (5 + 1e-6)]
    user_history = [1]
    @test predict_u2i(similar_items, similarity, user_history, 1, 1) == [2]
    @test predict_u2i(similar_items, similarity, user_history, 1, 2) == [2] # score=0 なので一個しか返らない
    @test predict_u2i(similar_items, similarity, user_history, 1, 1, drop_history = true) ==
          [2]
    @test predict_u2i(similar_items, similarity, user_history, 1, 2, drop_history = true) ==
          [2]
end

@testset "Predict item to item" begin
    similar_items = [2, 1]
    @test predict_i2i(similar_items, 1, 1, 1) == [2]
    @test predict_i2i(similar_items, 2, 1, 1) == [1]
    @test predict_i2i(similar_items, 1, 1, 2) == [2]
    @test predict_i2i(similar_items, 2, 1, 2) == [1]
end
