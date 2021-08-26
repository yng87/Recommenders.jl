abstract type AbstractMetric end

abstract type SingleListMetric <: AbstractMetric end

abstract type TopKMetric <: SingleListMetric end

struct Recall <: TopKMetric
    k::Int
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
        # result += (2^relevances[i] - 1) / log2(1.0 + i)
        result += (relevances[i]) / log2(1.0 + i)
    end
    return result
end

struct DCG <: TopKMetric
    k::Int
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

    n = min(length(ground_truth), k)
    idcg = _dcg(sort(true_relevance, rev = true)[1:n])

    return dcg / idcg
end

struct MeanMetric{T<:SingleListMetric} <: AbstractMetric
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

# old
# struct RecallAtK <: AbstractMetric
#     k::Int64
# end

# function (recall::RecallAtK)(recommends, ys)
#     rec = 0
#     for (pred, gt) in zip(recommends, ys)
#         if pred === nothing || length(pred) == 0
#             continue
#         end
#         k = min(recall.k, length(pred))
#         pred = pred[1:k]
#         num_hits = length(intersect(pred, gt))
#         rec += num_hits / length(gt)
#     end
#     return rec / length(ys)
# end

# struct PrecisionAtK <: AbstractMetric
#     k::Int64
# end

# function (precision::PrecisionAtK)(recommends, ys)
#     prec = 0
#     for (pred, gt) in zip(recommends, ys)
#         if pred === nothing || length(pred) == 0
#             continue
#         end
#         k = min(precision.k, length(pred))
#         pred = pred[1:k]
#         num_hits = length(intersect(pred, gt))
#         prec += num_hits / length(pred)
#     end
#     return prec / length(ys)
# end


# struct NDCG <: AbstractMetric
#     k::Int64
# end

# function (precision::NDCG)(recommends, ys)
#     ndcg = 0
#     for (pred, gt) in zip(recommends, ys)
#         if pred === nothing || length(pred) == 0
#             continue
#         end
#         k = min(precision.k, length(pred))
#         pred = pred[1:k]

#         dcg = 0
#         for i in eachindex(pred)
#             if pred[i] in gt
#                 dcg += 1.0 / log2(1.0 + i)
#             end
#         end

#         idcg = 0
#         for i = 1:min(length(pred), length(gt))
#             idcg += 1.0 / log2(1.0 + i)
#         end

#         ndcg += dcg / idcg
#     end
#     return ndcg / length(ys)
# end