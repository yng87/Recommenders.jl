module Recommender

export download, dataset, load

using HTTP, ZipFile, DataFrames, CSV
import Base: download

include("downloadutils.jl")
include("loadutils.jl")
include("dataset.jl")

end
