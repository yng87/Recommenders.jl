### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 786739dc-3b31-46f0-a99a-8d6b09c4e52c
begin
	using Pkg
	Pkg.activate("../Project.toml")
	using DataFrames, TableOperations, Tables, Random
end

# ╔═╡ 7e2731ee-b37b-4965-b17a-cfa958ad3dc7
using Recommender: Movielens100k, load_dataset, ratio_split, MostPopular, evaluate_u2i, PrecisionAtK, RecallAtK, NDCG

# ╔═╡ 386a8c7a-f5a6-4aff-96d1-a506edec8cfa
begin
	ml100k = Movielens100k()
	download(ml100k)
	rating, user, movie = load_dataset(ml100k)
end

# ╔═╡ c53e8744-2d82-4b13-85df-bb003792dc15
table = rating |> TableOperations.filter(x->Tables.getcolumn(x, :rating) >= 4);

# ╔═╡ fb96a4df-1b80-4a04-b2ef-ac302319fb8d
begin
	Random.seed!(1234);
	train_table, test_table = ratio_split(table, 0.8)
	length(Tables.rows(train_table)), length(Tables.rows(test_table))
end

# ╔═╡ 4fba58d6-1467-4aeb-92e9-c937091b5631
begin
	prec10 = PrecisionAtK(10)
	recall10 = RecallAtK(10)
	ndcg10 = NDCG(10)
	metrics = [prec10, recall10, ndcg10]
end

# ╔═╡ ace4f730-02a0-4c8f-805d-4c6d410836d1
model = MostPopular()

# ╔═╡ 06360c25-0069-4807-b691-949f5f5631cc
evaluate_u2i(model, train_table, test_table, metrics, 10, col_item=:movieid)

# ╔═╡ Cell order:
# ╠═786739dc-3b31-46f0-a99a-8d6b09c4e52c
# ╠═7e2731ee-b37b-4965-b17a-cfa958ad3dc7
# ╠═386a8c7a-f5a6-4aff-96d1-a506edec8cfa
# ╠═c53e8744-2d82-4b13-85df-bb003792dc15
# ╠═fb96a4df-1b80-4a04-b2ef-ac302319fb8d
# ╠═4fba58d6-1467-4aeb-92e9-c937091b5631
# ╠═ace4f730-02a0-4c8f-805d-4c6d410836d1
# ╠═06360c25-0069-4807-b691-949f5f5631cc
