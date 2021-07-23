using Test
using Recommender: RecallAtK

recommends = [[1, 2, 3], [4, 5], [], nothing]

ys = [[1], [2, 5], [4], [5]]

recall1 = RecallAtK(1)
@test recall1(recommends, ys) == 1 / 4

recall2 = RecallAtK(2)
@test recall2(recommends, ys) == (1 + 0.5) / 4
