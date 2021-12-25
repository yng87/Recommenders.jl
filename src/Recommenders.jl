module Recommenders

export AbstractRecommender,
    fit!,
    predict_u2i,
    predict_i2i,
    evaluate_u2i,
    save_model,
    load_model,
    MostPopular,
    ItemkNN,
    ImplicitMF,
    BPR,
    SLIM,
    Randomwalk,
    MeanDCG,
    MeanNDCG,
    MeanPrecision,
    MeanRecall,
    leave_one_out_split,
    ratio_split,
    EvaluateValidData

using HTTP,
    ZipFile,
    DataFrames,
    CSV,
    SparseArrays,
    Parameters,
    Random,
    Tables,
    TableOperations,
    Lasso,
    Distributions,
    JLD2
import Base: download

include("dataset/downloadutils.jl")
include("dataset/dataset.jl")
include("dataset/movielens.jl")
include("dataset/data_split.jl")
include("dataset/data_utils.jl")

include("core/item_knn.jl")
include("core/loss_function.jl")
include("core/randomwalk_bipartite.jl")
include("metric.jl")

include("model/base_recommender.jl")
include("model/utils.jl")
include("model/most_popular.jl")
include("model/item_knn.jl")
include("model/implicit_mf.jl")
include("model/bpr.jl")
include("model/slim.jl")
include("model/randomwalk.jl")


end
