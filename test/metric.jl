using Test
using Recommender: RecallAtK, PrecisionAtK, NDCG

@testset "Recall and precision" begin
    recommends = [[1, 2, 3], [4, 5], [], nothing]

    ys = [[1], [2, 5], [4], [5]]

    recall1 = RecallAtK(1)
    @test recall1(recommends, ys) == 1 / 4

    recall2 = RecallAtK(2)
    @test recall2(recommends, ys) == (1 + 0.5) / 4

    prec1 = PrecisionAtK(1)
    @test prec1(recommends, ys) == 1 / 4

    prec2 = PrecisionAtK(2)
    @test prec2(recommends, ys) == (0.5 + 0.5) / 4

end

@testset "NDCG" begin
    ndcg3 = NDCG(3)
    preds = [[1, 2, 4]]
    @test ndcg3(preds, [[1]]) == 1.0
    @test ndcg3(preds, [[2]]) == 1.0 / log2(1.0 + 2)
    @test ndcg3(preds, [[4]]) == 1.0 / log2(1.0 + 3)
    @test ndcg3(preds, [[1, 2]]) == 1.0
    @test ndcg3(preds, [[1, 4]]) ==
          (1.0 + 1.0 / log2(1.0 + 3)) / (1.0 + 1.0 / log2(1.0 + 2))

    ndcg2 = NDCG(2)
    preds = [[1, 2, 4], [5, 7, 9]]
    gts = [[1, 4], [7]]
    @test ndcg2(preds, gts) == (1.0 / (1.0 + 1.0 / log2(1.0 + 2)) + 1.0 / log2(1.0 + 2)) / 2
    @test ndcg3(preds, gts) ==
          (
        (1.0 + 1.0 / log2(1.0 + 3)) / (1.0 + 1.0 / log2(1.0 + 2)) + 1.0 / log2(1.0 + 2)
    ) / 2
end
