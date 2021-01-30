module Recommender

export download, dataset, load

using HTTP, ZipFile, DataFrames, CSV, SparseArrays, MLJ, MLJBase, Parameters
import Base: download

include("downloadutils.jl")
include("loadutils.jl")
include("dataset.jl")

include("ItemkNN.jl")

end
