using DataFrames, TableOperations, Tables, Random
using TreeParzen
using Recommenders:
    Movielens1M,
    load_dataset,
    ratio_split,
    ImplicitMF,
    evaluate_u2i,
    MeanPrecision,
    MeanRecall,
    MeanNDCG

function one_write(filepath, params, result)
    n_epochs = convert(Int, params[:n_epochs])
    n_negatives = convert(Int, params[:n_negatives])
    learning_rate = params[:learning_rate]
    dimension = convert(Int, 2^params[:log2_dimension])
    reg_coeff = params[:reg_coeff]

    ndcg10 = result[:ndcg10]
    precision10 = result[:precision10]
    recall10 = result[:recall10]

    str = "$n_epochs,$n_negatives,$learning_rate,$dimension,$reg_coeff,$ndcg10,$precision10,$recall10\n"

    open(filepath, "a") do io
        write(io, str)
    end
end


function search(filepath)
    ml1M = Movielens1M()
    download(ml1M)
    rating, _, _ = load_dataset(ml1M)

    Random.seed!(1234)
    train_valid_table, test_table = ratio_split(rating, 0.8)
    train_table, valid_table = ratio_split(train_valid_table, 0.8)

    prec10 = MeanPrecision(10)
    recall10 = MeanRecall(10)
    ndcg10 = MeanNDCG(10)
    metrics = [prec10, recall10, ndcg10]

    function invert_output(params)
        @info params
        n_epochs = convert(Int, params[:n_epochs])
        n_negatives = convert(Int, params[:n_negatives])
        learning_rate = params[:learning_rate]
        dimension = convert(Int, 2^params[:log2_dimension])
        reg_coeff = params[:reg_coeff]

        model = ImplicitMF(dimension, true, reg_coeff)

        result = evaluate_u2i(
            model,
            train_table,
            valid_table,
            metrics,
            10,
            col_item = :movieid,
            n_epochs = n_epochs,
            n_negatives = n_negatives,
            learning_rate = learning_rate,
            drop_history = true,
            verbose = 16,
        )
        @info result
        one_write(filepath, params, result)
        return -result[:ndcg10]
    end


    open(filepath, "w") do io
        header = "#n_epochs,n_negatives,learning_rate,dimension,reg_coeff,ndcg10,precision10,recall10\n"
        write(io, header)
    end

    @info "coarse search"
    best_result = Inf
    best_params = nothing
    for lr in [1e-3, 3e-3, 1e-2], n_neg in [4, 8, 16]
        params = Dict(
            :n_epochs => 128,
            :n_negatives => n_neg,
            :learning_rate => lr,
            :log2_dimension => 6,
            :reg_coeff => 1e-20,
        )
        result = invert_output(params)
        if result < best_result
            best_result = result
            best_params = params
        end
    end

    for reg_coeff in [1e-3, 3e-3, 1e-2]
        params = copy(best_params)
        params[:reg_coeff] = reg_coeff
        result = invert_output(params)
        if result < best_result
            best_result = result
            best_params = params
        end
    end

    @info "refinement"
    space = Dict(
        :n_epochs =>
            HP.Choice(:n_epochs, [best_params[:n_epochs], best_params[:n_epochs] * 2]),
        :n_negatives => HP.QuantUniform(
            :n_negatives,
            best_params[:n_negatives] - 2.0,
            best_params[:n_negatives] + 2.0,
            1.0,
        ),
        :learning_rate => HP.Uniform(
            :learning_rate,
            best_params[:learning_rate] * 0.5,
            best_params[:learning_rate] * 2.0,
        ),
        :log2_dimension => HP.Choice(:log2_dimension, [6, 7, 8]),
        :reg_coeff => HP.Uniform(
            :reg_coeff,
            best_params[:reg_coeff] * 0.5,
            best_params[:reg_coeff] * 2.0,
        ),
    )

    best = fmin(invert_output, space, 20, logging_interval = -1)
    @info best

    @info "Evaluate best model."

    best_model = ImplicitMF(convert(Int, 2^best[:log2_dimension]), true, best[:reg_coeff])
    result = evaluate_u2i(
        best_model,
        train_valid_table,
        test_table,
        metrics,
        10,
        col_item = :movieid,
        n_epochs = convert(Int, best[:n_epochs]),
        n_negatives = convert(Int, best[:n_negatives]),
        learning_rate = best[:learning_rate],
        drop_history = true,
        verbose = 16,
    )
    @info result

end

search("./log_param_search_mf_ml1M.txt")