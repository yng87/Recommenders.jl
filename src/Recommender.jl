module Recommender

using TableOperations: include
export download, load, kNNRecommender

# metrics
export hitrate

using HTTP,
    ZipFile,
    DataFrames,
    CSV,
    SparseArrays,
    MLJ,
    MLJBase,
    Parameters,
    Tables,
    TableOperations
import Base: download

include("dataset/downloadutils.jl")
include("dataset/dataset.jl")
include("dataset/movielens.jl")

include("model/BaseRecommender.jl")
include("model/kNNRecommender.jl")

include("evaluate.jl")


end
