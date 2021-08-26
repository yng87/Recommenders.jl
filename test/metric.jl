using Test
using Recommender: Recall, Precision, DCG, NDCG, MeanRecall, MeanPrecision

@testset "Recall and precision" begin
    recall1 = Recall(1)

    recommends = [[1, 2, 3], [4, 5], [], nothing]
    ys = [[1], [2, 5], [4], [5]]

    expected_recall1 = [1.0, 0.0, 0.0, 0.0]
    expected_precision1 = [1.0, 0.0, 0.0, 0.0]
    expected_recall2 = [1.0, 0.5, 0.0, 0.0]
    expected_precision2 = [0.5, 0.5, 0.0, 0.0]

    recall1 = Recall(1)
    recall2 = Recall(2)
    precision1 = Precision(1)
    precision2 = Precision(2)

    for i in eachindex(recommends)
        @test recall1(recommends[i], ys[i]) == expected_recall1[i]
        @test recall2(recommends[i], ys[i]) == expected_recall2[i]
        @test precision1(recommends[i], ys[i]) == expected_precision1[i]
        @test precision2(recommends[i], ys[i]) == expected_precision2[i]
    end


    # mean_recall1 = MeanMetric(Recall(1))
    mean_recall1 = MeanRecall(1)
    @test mean_recall1(recommends, ys) == 1 / 4

    mean_recall12 = MeanRecall(2)
    @test mean_recall12(recommends, ys) == (1 + 0.5) / 4

    mean_prec1 = MeanPrecision(1)
    @test mean_prec1(recommends, ys) == 1 / 4

    mean_prec2 = MeanPrecision(2)
    @test mean_prec2(recommends, ys) == (0.5 + 0.5) / 4

end

@testset "DCG" begin
    dcg1 = DCG(1)
    dcg2 = DCG(2)
    dcg3 = DCG(3)
end

# @testset "NDCG" begin
#     ndcg3 = NDCG(3)
#     preds = [[1, 2, 4]]
#     @test ndcg3(preds, [[1]]) == 1.0
#     @test ndcg3(preds, [[2]]) == 1.0 / log2(1.0 + 2)
#     @test ndcg3(preds, [[4]]) == 1.0 / log2(1.0 + 3)
#     @test ndcg3(preds, [[1, 2]]) == 1.0
#     @test ndcg3(preds, [[1, 4]]) ==
#           (1.0 + 1.0 / log2(1.0 + 3)) / (1.0 + 1.0 / log2(1.0 + 2))

#     ndcg2 = NDCG(2)
#     preds = [[1, 2, 4], [5, 7, 9]]
#     gts = [[1, 4], [7]]
#     @test ndcg2(preds, gts) == (1.0 / (1.0 + 1.0 / log2(1.0 + 2)) + 1.0 / log2(1.0 + 2)) / 2
#     @test ndcg3(preds, gts) ==
#           (
#         (1.0 + 1.0 / log2(1.0 + 3)) / (1.0 + 1.0 / log2(1.0 + 2)) + 1.0 / log2(1.0 + 2)
#     ) / 2
# end
