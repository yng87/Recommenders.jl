### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ fa37fec4-0256-11ec-04c8-bbcd904fd2be
begin
    using Pkg
    Pkg.activate("../Project.toml")
    using DataFrames, TableOperations, Tables, Random, TreeParzen
end

# ╔═╡ add3edb5-6024-4034-941a-7dc40955dc40
using Recommender:
    Movielens1M,
    load_dataset,
    ratio_split,
    ImplicitMF,
    evaluate_u2i,
    PrecisionAtK,
    RecallAtK,
    NDCG

# ╔═╡ b7110972-4f70-4351-bbdb-a1a14f9b4516
begin
    ml1M = Movielens1M()
    download(ml1M)
    rating, user, movie = load_dataset(ml1M)
end

# ╔═╡ d9fb81d7-d9da-45cb-8886-8c5fad642418
table = rating |> TableOperations.filter(x->Tables.getcolumn(x, :rating) >= 4);

# ╔═╡ 82aa8fd7-f9b6-4441-b7f2-7d0e75dd2174
begin
    Random.seed!(1234);
	train_valid_table, test_table = ratio_split(table, 0.8)

	train_table, valid_table = ratio_split(train_valid_table, 0.8)
	length(Tables.rows(train_table)), length(Tables.rows(valid_table)), 	length(Tables.rows(test_table))
end

# ╔═╡ 3f3e299b-c3a0-41da-ba6c-f62cbbf1b88f
begin
    prec10 = PrecisionAtK(10)
    recall10 = RecallAtK(10)
    ndcg10 = NDCG(10)
    metrics = [prec10, recall10, ndcg10]
end

# ╔═╡ 56e2aea0-2b9c-4e47-80e8-b3551d03bf14
model = ImplicitMF(128, true, 0.005)

# ╔═╡ 464b3156-4207-4913-b52f-caf840c894ff
evaluate_u2i(
	model, 
	train_valid_table,
	test_table, 
	metrics, 
	10, 
	col_item=:movieid, 
	n_epochs=256, 
	n_negatives=8, 
	learning_rate=0.002, 
	drop_history=true
)

# ╔═╡ Cell order:
# ╠═fa37fec4-0256-11ec-04c8-bbcd904fd2be
# ╠═add3edb5-6024-4034-941a-7dc40955dc40
# ╠═b7110972-4f70-4351-bbdb-a1a14f9b4516
# ╠═d9fb81d7-d9da-45cb-8886-8c5fad642418
# ╠═82aa8fd7-f9b6-4441-b7f2-7d0e75dd2174
# ╠═3f3e299b-c3a0-41da-ba6c-f62cbbf1b88f
# ╠═56e2aea0-2b9c-4e47-80e8-b3551d03bf14
# ╠═464b3156-4207-4913-b52f-caf840c894ff
