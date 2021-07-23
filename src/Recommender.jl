module Recommender

using TableOperations: include
export download, load_dataset, retrieve, ItemkNN

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
    TableOperations,
    Random
import Base: download
import MLJModelInterface
const MMI = MLJModelInterface

include("dataset/downloadutils.jl")
include("dataset/dataset.jl")
include("dataset/movielens.jl")

include("data_split.jl")

include("model/extended_operations.jl")
include("model/item_knn.jl")

include("evaluate.jl")


end
