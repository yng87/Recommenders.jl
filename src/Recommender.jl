module Recommender

export download, dataset

using HTTP, ZipFile, DataFrames, CSV
import Base: download

include("Dataset.jl")

end
