### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ d185de30-024c-11ec-12e7-13aa20669d9e
begin
    using Pkg
    Pkg.activate("../Project.toml")
    using DataFrames, TableOperations, Tables, Random
end

# ╔═╡ ac1c7fab-a7e7-4d29-87b3-7c63d4c3521a
using Recommender:
    Movielens1M,
    load_dataset,
    ratio_split,
    MostPopular,
    evaluate_u2i,
    PrecisionAtK,
    RecallAtK,
    NDCG

# ╔═╡ 320a8c6d-1b85-4dcb-8b71-5ab51f8411f6
begin
    ml1M = Movielens1M()
    download(ml1M)
    rating, user, movie = load_dataset(ml1M)
end

# ╔═╡ f32cabfb-69ea-4404-a699-c46578d9ace6
table = rating |> TableOperations.filter(x->Tables.getcolumn(x, :rating) >= 4);

# ╔═╡ 42b4cca4-90e7-4bd5-808e-efab9fd48089
begin
    Random.seed!(1234)
    train_table, test_table = ratio_split(table, 0.8)
    length(Tables.rows(train_table)), length(Tables.rows(test_table))
end

# ╔═╡ bd44aa09-1b25-482c-a242-103c8d6f42c4
begin
    prec10 = PrecisionAtK(10)
    recall10 = RecallAtK(10)
    ndcg10 = NDCG(10)
    metrics = [prec10, recall10, ndcg10]
end

# ╔═╡ 5ed7b6b9-ce37-4eed-8db2-fffcad3f72fe
model = MostPopular()

# ╔═╡ 969329c3-5dcc-43b9-87da-f3629dadf486
evaluate_u2i(model, train_table, test_table, metrics, 10, col_item = :movieid)

# ╔═╡ Cell order:
# ╠═d185de30-024c-11ec-12e7-13aa20669d9e
# ╠═ac1c7fab-a7e7-4d29-87b3-7c63d4c3521a
# ╠═320a8c6d-1b85-4dcb-8b71-5ab51f8411f6
# ╠═f32cabfb-69ea-4404-a699-c46578d9ace6
# ╠═42b4cca4-90e7-4bd5-808e-efab9fd48089
# ╠═bd44aa09-1b25-482c-a242-103c8d6f42c4
# ╠═5ed7b6b9-ce37-4eed-8db2-fffcad3f72fe
# ╠═969329c3-5dcc-43b9-87da-f3629dadf486
