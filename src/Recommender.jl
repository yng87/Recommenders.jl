module Recommender

using TableOperations: include

using HTTP,
    ZipFile,
    DataFrames,
    CSV,
    SparseArrays,
    MLJ,
    MLJBase,
    Parameters,
    Tables,
    TableOperations,
    Random
import Base: download
import MLJModelInterface
const MMI = MLJModelInterface

include("dataset/downloadutils.jl")
include("dataset/dataset.jl")
include("dataset/movielens.jl")
include("dataset/data_split.jl")
include("dataset/data_utils.jl")

include("algorithm/item_knn.jl")

include("metric.jl")


end
