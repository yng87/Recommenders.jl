module Recommender

export download, dataset, load, ItemkNN

using HTTP, ZipFile, DataFrames, CSV, SparseArrays, MLJ, MLJBase, Parameters, Tables, TableOperations
import Base: download

include("downloadutils.jl")
include("loadutils.jl")
include("dataset.jl")
include("evaluate.jl")

include("kNNRecommender.jl")

end
