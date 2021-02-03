module Recommender

export download, dataset, load, ItemkNN

using HTTP, ZipFile, DataFrames, CSV, SparseArrays, MLJ, MLJBase, Parameters, Tables, TableOperations
import Base: download

include("base_predict.jl")

include("downloadutils.jl")
include("loadutils.jl")
include("dataset.jl")

include("ItemkNN.jl")

end
