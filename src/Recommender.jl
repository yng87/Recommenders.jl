module Recommender

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

include("evaluate.jl")

include("kNNRecommender.jl")

end
