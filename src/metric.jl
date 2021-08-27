abstract type AbstractMetric end

abstract type SingleRecommendMetric <: AbstractMetric end

abstract type TopKMetric <: SingleRecommendMetric end

struct Recall <: TopKMetric
    k::Int
    name::AbstractString

    Recall(k::Int) = new(k, "recall")
end

function (recall::Recall)(recommend_list, ground_truth)
    if recommend_list === nothing || length(recommend_list) == 0
        return 0.0
    end

    k = min(recall.k, length(recommend_list))
    truncated_recom = recommend_list[1:k]
    num_hits = length(intersect(truncated_recom, ground_truth))
    return num_hits / length(ground_truth)
end

struct Precision <: TopKMetric
    k::Int
    name::AbstractString

    Precision(k::Int) = new(k, "precision")
end

function (precision::Precision)(recommend_list, ground_truth)
    if recommend_list === nothing || length(recommend_list) == 0
        return 0.0
    end

    k = min(precision.k, length(recommend_list))
    truncated_recom = recommend_list[1:k]
    num_hits = length(intersect(truncated_recom, ground_truth))
    return num_hits / length(truncated_recom)
end

function _dcg(relevances)
    result = 0.0
    for i in eachindex(relevances)
        result += (2^relevances[i] - 1) / log2(1.0 + i)
    end
    return result
end

struct DCG <: TopKMetric
    k::Int
    name::AbstractString

    DCG(k::Int) = new(k, "dcg")
end

function (dcg::DCG)(recommend_list, ground_truth, true_relevance = nothing)
    if recommend_list === nothing || length(recommend_list) == 0
        return 0.0
    end

    k = min(dcg.k, length(recommend_list))
    truncated_recom = recommend_list[1:k]

    if true_relevance === nothing
        true_relevance = ones(length(ground_truth))
    end

    relevances = zeros(k)
    for i = 1:k
        j = findfirst(truncated_recom[i] .== ground_truth)
        if !(j === nothing)
            relevances[i] = true_relevance[j]
        end
    end
    return _dcg(relevances)
end

struct NDCG <: TopKMetric
    k::Int
    name::AbstractString

    NDCG(k::Int) = new(k, "ndcg")
end

function (ndcg::NDCG)(recommend_list, ground_truth, true_relevance = nothing)
    if recommend_list === nothing || length(recommend_list) == 0
        return 0.0
    end

    k = min(ndcg.k, length(recommend_list))
    truncated_recom = recommend_list[1:k]

    if true_relevance === nothing
        true_relevance = ones(length(ground_truth))
    end

    relevances = zeros(k)
    for i = 1:k
        j = findfirst(truncated_recom[i] .== ground_truth)
        if !(j === nothing)
            relevances[i] = true_relevance[j]
        end
    end
    dcg = _dcg(relevances)

    # Note: Table 3 of [Dacrema, Cremonesi and Jannach, 2019]("Are we really...")
    # probably adopts n = length(true_relevance) for IDCG.
    # I believe it is not suitable and use instead the definition below
    # This formulation is seen, for instance, in
    # [Rendle 2021]("Item Recommendation from Implicit Feedback")
    n = min(length(ground_truth), k)
    idcg = _dcg(sort(true_relevance, rev = true)[1:n])

    return dcg / idcg
end

#only for implicit recommendation
struct MeanMetric{T<:SingleRecommendMetric} <: AbstractMetric
    base_metric::T
end

function (metric::MeanMetric)(recommend_lists, ground_truth_list)
    result = 0.0
    for (recom, gt) in zip(recommend_lists, ground_truth_list)
        result += metric.base_metric(recom, gt)
    end
    return result / length(recommend_lists)
end

MeanRecall(k) = MeanMetric(Recall(k))
MeanPrecision(k) = MeanMetric(Precision(k))
MeanDCG(k) = MeanMetric(DCG(k))
MeanNDCG(k) = MeanMetric(NDCG(k))

name(metric::MeanMetric) = "$(metric.base_metric.name)$(metric.base_metric.k)"

