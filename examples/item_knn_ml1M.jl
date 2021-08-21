### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 9a16e318-0254-11ec-206c-0d3cc3903f69
begin
    using Pkg
    Pkg.activate("../Project.toml")
    using DataFrames, TableOperations, Tables, Random, TreeParzen
end

# ╔═╡ 0baa9c88-995c-4e3b-a557-d14b66efb473
using Recommender:
    Movielens1M,
    load_dataset,
    ratio_split,
    ItemkNN,
    evaluate_u2i,
    PrecisionAtK,
    RecallAtK,
    NDCG

# ╔═╡ d977eda1-7b43-4fa2-bd1e-28933bb571e1
begin
    ml1M = Movielens1M()
    download(ml1M)
    rating, user, movie = load_dataset(ml1M)
end

# ╔═╡ 7cbabbcd-77fc-4d66-8a41-06d178a532f0
table = rating |> TableOperations.filter(x->Tables.getcolumn(x, :rating) >= 4);

# ╔═╡ 161033d6-716d-43fe-8fde-61ba18e72cb2
begin
    Random.seed!(1234);
	train_valid_table, test_table = ratio_split(table, 0.8)

	train_table, valid_table = ratio_split(train_valid_table, 0.8)
	length(Tables.rows(train_table)), length(Tables.rows(valid_table)), 	length(Tables.rows(test_table))
end

# ╔═╡ a4329e03-16e9-4003-9260-fd2ebcfbe06e
begin
    prec10 = PrecisionAtK(10)
    recall10 = RecallAtK(10)
    ndcg10 = NDCG(10)
    metrics = [prec10, recall10, ndcg10]
end

# ╔═╡ 112841e4-cedc-4ad8-8e13-21ca9384246b
space = Dict(
    :topk=>HP.QuantUniform(:topk, 10., 500., 1.),
    :shrink=>HP.LogUniform(:shrink, log(1e-3), log(1e3)),
    :weighting=>HP.Choice(:weighting, 
        [
            Dict(:weighting=>:dummy, :weighting_at_inference=>false),
            Dict(:weighting=>:tfidf, :weighting_at_inference=>false),
            Dict(:weighting=>:bm25, 
				:weighting_at_inference=>HP.Choice(
					:weighting_at_inference, [true, false]))
        ]
    ),
    :normalize=>HP.Choice(:normalize, [true, false])
)

# ╔═╡ 00ff7619-d97d-4b30-9e7f-e8587e4f1050
function invert_output(params)
    k = convert(Int, params[:topk])
    model = ItemkNN(
		k, 
		params[:shrink],
		params[:weighting][:weighting],
		params[:weighting][:weighting_at_inference],params[:normalize]
	)
    result = evaluate_u2i(
		model, 
		train_table, 
		valid_table, 
		metrics, 
		10, 
		col_user=:userid, 
		col_item=:movieid,
		col_rating=:rating, 
		drop_history=false
	)
    @info params, result
    return -result[end]
end

# ╔═╡ bd765fb7-3940-4ad4-8f3b-f9d6fb002ed6
best = fmin(invert_output, space, 10, logging_interval=-1)

# ╔═╡ 4f52c8fc-7ece-44a5-bc2a-b4fb97595e59
best_model = ItemkNN(
	convert(Int, best[:topk]), 
	best[:shrink],
	best[:weighting][:weighting],
	best[:weighting][:weighting_at_inference],
	best[:normalize]
)

# ╔═╡ 4d56d08c-8272-46c0-8b0a-e524b9e28053
evaluate_u2i(
	best_model, 
	train_valid_table, 
	test_table, 
	metrics, 
	10, 
	col_user=:userid, 
	col_item=:movieid, 
	col_rating=:rating, 
	drop_hisotory=false
)

# ╔═╡ Cell order:
# ╠═9a16e318-0254-11ec-206c-0d3cc3903f69
# ╠═0baa9c88-995c-4e3b-a557-d14b66efb473
# ╠═d977eda1-7b43-4fa2-bd1e-28933bb571e1
# ╠═7cbabbcd-77fc-4d66-8a41-06d178a532f0
# ╠═161033d6-716d-43fe-8fde-61ba18e72cb2
# ╠═a4329e03-16e9-4003-9260-fd2ebcfbe06e
# ╠═112841e4-cedc-4ad8-8e13-21ca9384246b
# ╠═00ff7619-d97d-4b30-9e7f-e8587e4f1050
# ╠═bd765fb7-3940-4ad4-8f3b-f9d6fb002ed6
# ╠═4f52c8fc-7ece-44a5-bc2a-b4fb97595e59
# ╠═4d56d08c-8272-46c0-8b0a-e524b9e28053
