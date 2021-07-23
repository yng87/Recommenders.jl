using Test, DataFrames, MLJ
using Recommender: LeaveOneOut, train_test_pairs

X = (
    userid = [1, 1, 1, 2, 2, 3, 3, 3, 3],
    itemid = [1, 2, 3, 1, 4, 2, 4, 5, 6],
    timestamp = [0, 2, 10, 3, 5, 4, 5, 11, -1],
)

loo = LeaveOneOut(time_column = :timestamp, key_column = :userid)
X_rows = 1:9

train, test = train_test_pairs(loo, X_rows, X)[1]

@test train == [1, 2, 4, 6, 7, 9]
@test test == [3, 5, 8]
