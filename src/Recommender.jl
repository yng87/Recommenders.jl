module Recommender

export download, dataset, load

using HTTP, ZipFile, DataFrames, CSV
import Base: download

include("Dataset.jl")

end
