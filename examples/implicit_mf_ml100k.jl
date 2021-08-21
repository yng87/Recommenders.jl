### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 4948045c-0255-11ec-3409-cbe31bf1f07c
begin
    using Pkg
    Pkg.activate("../Project.toml")
    using DataFrames, TableOperations, Tables, Random, TreeParzen
end

# ╔═╡ cdcde31f-7181-49c0-a337-d06876ee53d0
using Recommender:
    Movielens100k,
    load_dataset,
    ratio_split,
    ImplicitMF,
    evaluate_u2i,
    PrecisionAtK,
    RecallAtK,
    NDCG

# ╔═╡ bab56522-4e1a-4e82-9165-ebb11b42903c
begin
	ml100k = Movielens100k()
	download(ml100k)
	rating, user, movie = load_dataset(ml100k);
end

# ╔═╡ 47cf35ea-b2b9-4a50-85bf-17aa22fc420a
table = rating |> TableOperations.filter(x->Tables.getcolumn(x, :rating) >= 4);

# ╔═╡ ddc7669b-5b1b-40e2-ac4c-96741d40b894
begin
	Random.seed!(1234);
	train_valid_table, test_table = ratio_split(table, 0.8)

	train_table, valid_table = ratio_split(train_valid_table, 0.8)
	length(Tables.rows(train_table)), length(Tables.rows(valid_table)), length(Tables.rows(test_table))
end

# ╔═╡ 042cc00b-dc64-42e6-88cc-f84a38793760
begin
	prec10 = PrecisionAtK(10)
	recall10 = RecallAtK(10)
	ndcg10 = NDCG(10)
	metrics = [prec10, recall10, ndcg10]
end

# ╔═╡ 30c63a61-f17e-4002-a00d-7bed5f1f3aba
model = ImplicitMF(128, true, 0.005)

# ╔═╡ 10cc9d11-2043-46a2-b869-6377a3f93152
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
# ╠═4948045c-0255-11ec-3409-cbe31bf1f07c
# ╠═cdcde31f-7181-49c0-a337-d06876ee53d0
# ╠═bab56522-4e1a-4e82-9165-ebb11b42903c
# ╠═47cf35ea-b2b9-4a50-85bf-17aa22fc420a
# ╠═ddc7669b-5b1b-40e2-ac4c-96741d40b894
# ╠═042cc00b-dc64-42e6-88cc-f84a38793760
# ╠═30c63a61-f17e-4002-a00d-7bed5f1f3aba
# ╠═10cc9d11-2043-46a2-b869-6377a3f93152
