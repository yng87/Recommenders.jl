module Recommender

using TableOperations: include
export download, load_all, ItemkNN

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
import MLJModelInterface
const MMI = MLJModelInterface

include("dataset/downloadutils.jl")
include("dataset/dataset.jl")
include("dataset/movielens.jl")

include("model/BaseRecommender.jl")
include("model/item_knn.jl")

include("evaluate.jl")


end
