module Recommender


using HTTP,
    ZipFile, DataFrames, CSV, SparseArrays, Parameters, Random, Tables, TableOperations
import Base: download

include("dataset/downloadutils.jl")
include("dataset/dataset.jl")
include("dataset/movielens.jl")
include("dataset/data_split.jl")
include("dataset/data_utils.jl")

include("core/item_knn.jl")
include("core/loss_function.jl")
include("metric.jl")

include("model/base_recommender.jl")
include("model/most_popular.jl")
include("model/item_knn.jl")
include("model/implicit_mf.jl")
include("model/bpr.jl")



end
