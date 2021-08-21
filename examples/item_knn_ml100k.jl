### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ e8a9671e-0252-11ec-3b75-3d802c7fc5ba
begin
    using Pkg
    Pkg.activate("../Project.toml")
    using DataFrames, TableOperations, Tables, Random, TreeParzen
end

# ╔═╡ 6c220334-62f0-4ab3-9461-0b48b0210468
using Recommender:
    Movielens100k,
    load_dataset,
    ratio_split,
    ItemkNN,
    evaluate_u2i,
    PrecisionAtK,
    RecallAtK,
    NDCG

# ╔═╡ 13a0080c-5a73-4e3e-8b09-50028aaef541
begin
    ml100k = Movielens100k()
    download(ml100k)
    rating, user, movie = load_dataset(ml100k)
end

# ╔═╡ 54e0080d-fe53-4000-a92c-03b5ee9b780f
table = rating |> TableOperations.filter(x->Tables.getcolumn(x, :rating) >= 4);

# ╔═╡ 5db4ac14-849b-4568-b1d9-91f639f819bd
begin
    Random.seed!(1234);
	train_valid_table, test_table = ratio_split(table, 0.8)

	train_table, valid_table = ratio_split(train_valid_table, 0.8)
	length(Tables.rows(train_table)), length(Tables.rows(valid_table)), 	length(Tables.rows(test_table))
end

# ╔═╡ d3ea8e30-f446-4b1d-947f-96d0693da937
begin
    prec10 = PrecisionAtK(10)
    recall10 = RecallAtK(10)
    ndcg10 = NDCG(10)
    metrics = [prec10, recall10, ndcg10]
end

# ╔═╡ 34cdec77-31c3-4e50-9ccf-b771502ed94d
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

# ╔═╡ 53482ea3-3f60-4f23-81f5-1c40b8d8bb5d
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

# ╔═╡ e29bf4c8-7bd5-4598-9c8f-9356baee2619
best = fmin(invert_output, space, 20, logging_interval=-1)

# ╔═╡ 68fdafab-6669-49df-a27d-95db8e3ebcfe
best_model = ItemkNN(
	convert(Int, best[:topk]), 
	best[:shrink],
	best[:weighting][:weighting],
	best[:weighting][:weighting_at_inference],
	best[:normalize]
)

# ╔═╡ 72f0340e-74c0-4b43-ad9e-dc56ffbfcf55
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
# ╠═e8a9671e-0252-11ec-3b75-3d802c7fc5ba
# ╠═6c220334-62f0-4ab3-9461-0b48b0210468
# ╠═13a0080c-5a73-4e3e-8b09-50028aaef541
# ╠═54e0080d-fe53-4000-a92c-03b5ee9b780f
# ╠═5db4ac14-849b-4568-b1d9-91f639f819bd
# ╠═d3ea8e30-f446-4b1d-947f-96d0693da937
# ╠═34cdec77-31c3-4e50-9ccf-b771502ed94d
# ╠═53482ea3-3f60-4f23-81f5-1c40b8d8bb5d
# ╠═e29bf4c8-7bd5-4598-9c8f-9356baee2619
# ╠═68fdafab-6669-49df-a27d-95db8e3ebcfe
# ╠═72f0340e-74c0-4b43-ad9e-dc56ffbfcf55
