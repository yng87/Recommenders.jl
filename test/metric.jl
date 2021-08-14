using Test
using Recommender: RecallAtK, PrecisionAtK

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