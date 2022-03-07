var documenterSearchIndex = {"docs":
[{"location":"metrics/#Evaluation-metrics","page":"Metrics","title":"Evaluation metrics","text":"","category":"section"},{"location":"metrics/","page":"Metrics","title":"Metrics","text":"The performance measure of recommendation compares the each recommended item list to the ground truth items, and averages them over whole recommendation list. Here is an example","category":"page"},{"location":"metrics/","page":"Metrics","title":"Metrics","text":"# Two recommendation, each of which has a collection of predicted item ids with descending order of scores.\nrecommends = [\n    [1, 2],\n    [4, 5]\n]\n# Ground truth item ids corresponding to each recommendation\nground_truth = [\n    [1],\n    [4, 5]\n]","category":"page"},{"location":"metrics/","page":"Metrics","title":"Metrics","text":"The Precision@2 for the first entry is 12=05, while the second is 22=1. Therefore, the mean Precision@2 is 075.","category":"page"},{"location":"metrics/","page":"Metrics","title":"Metrics","text":"In Recommenders.jl, this computation is done by","category":"page"},{"location":"metrics/","page":"Metrics","title":"Metrics","text":"Recommenders: MeanPrecision\nprec2 = MeanPrecision(2) # metrics are implemented as callable struct\nprec2(recommends, ground_truth)\n# 0.75","category":"page"},{"location":"metrics/","page":"Metrics","title":"Metrics","text":"Currently the following metrics are implemented. They are all descendent of MeanMetric type.","category":"page"},{"location":"metrics/","page":"Metrics","title":"Metrics","text":"Modules = [Recommenders]\nOrder   = [:type, :function]\nPages   = [\"metric.jl\"]","category":"page"},{"location":"metrics/#Recommenders.MeanDCG-Tuple{Any}","page":"Metrics","title":"Recommenders.MeanDCG","text":"MeanDCG(k)\n\nCreate callbale struct to compute DCG@k averaged over all predictions. DCG@k is defined by\n\nmathrmDCGk = sum_i=1^mathrmmin(k mathrmlength(textprediction)) frac2^r_i-1log(i+1)\n\nwhere r_i is the true relevance for the i-th predicted item (binary for implicit feedback).\n\nExample\n\ndcg10 = MeanDCG(10)\ndcg10(predictions, ground_truth)\n\n\n\n\n\n","category":"method"},{"location":"metrics/#Recommenders.MeanNDCG-Tuple{Any}","page":"Metrics","title":"Recommenders.MeanNDCG","text":"MeanNDCG(k)\n\nCreate callbale struct to compute NDCG@k averaged over all predictions. NDCG@k is defined by\n\nmathrmNDCGk = fracmathrmDCGkmathrmIDCGk\n\nwhere IDCG is the ideal DCG, prediction sorted by true relevance. Note that if the number of ground truth items is smaller than k, the predicted item list is truncated to that length.\n\nExample\n\nndcg10 = MeanNDCG(10)\nndcg10(predictions, ground_truth)\n\n\n\n\n\n","category":"method"},{"location":"metrics/#Recommenders.MeanPrecision-Tuple{Any}","page":"Metrics","title":"Recommenders.MeanPrecision","text":"MeanPrecision(k)\n\nCreate callbale struct to compute Precision@k averaged over all predictions. Precision@k is defined by\n\nmathrmPrecisionk = frac(textground truth) cap (texttop k prediction)k\n\nExample\n\nprec10 = MeanPrecision(10)\nprec10(predictions, ground_truth)\n\n\n\n\n\n","category":"method"},{"location":"metrics/#Recommenders.MeanRecall-Tuple{Any}","page":"Metrics","title":"Recommenders.MeanRecall","text":"MeanRecall(k)\n\nCreate callbale struct to compute Recall@k averaged over all predictions. Recall@k is defined by\n\nmathrmRecallk = frac(textground truth) cap (texttop k prediction)(textground truth)\n\nExample\n\nrecall10 = MeanRecall(10)\nrecall10(predictions, ground_truth)\n\n\n\n\n\n","category":"method"},{"location":"getting_started/#Getting-started","page":"Getting started","title":"Getting started","text":"","category":"section"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"We show how to train a matrix factorization model on classic Movielens 100k dataset. Although this is a explicit feedback data with explicit user rating on movies, we treat it as implicit feedback where all the (user, movie) pairs in the dataset are regarded as positive (label=1) while the other pairs are negative (label=0).","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"One needs to download the dataset from official page, and extract  data to any location you like.","category":"page"},{"location":"getting_started/#Data-preparation","page":"Getting started","title":"Data preparation","text":"","category":"section"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"Recommenders.jl assumes tabular data as input. To handle them, we use Tables.jl - abstract interface to handle all kinds of tabular objects.","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"CSV data, created by CSV.jl, is one such table object. Let's load movie rating data by using CSV.File:","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"using CSV\nrating = CSV.File(\n    joinpath(<path/to/movielens100k>, \"u.data\"),\n    delim = \"\\t\",\n    header = [:userid, :movieid, :rating, :timestamp],\n)","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"rating looks like","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"100000-element CSV.File{false}:\n CSV.Row: (userid = 196, movieid = 242, rating = 3, timestamp = 881250949)\n CSV.Row: (userid = 186, movieid = 302, rating = 3, timestamp = 891717742)\n CSV.Row: (userid = 22, movieid = 377, rating = 1, timestamp = 878887116)\n ⋮","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"For clarity, let's replace movie ids by their titles:","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"using Tables, TableOperations\n\nmovie = CSV.File(\n    joinpath(<path/to/movielens100k>, \"u.item\"),\n    delim = \"|\",\n    header = [\n        :movieid,\n        :movie_title,\n        :release_date,\n        :video_release_date,\n        :IMDbURL,\n        :unknown,\n        :Action,\n        :Adventure,\n        :Animation,\n        :Childrens,\n        :Comedy,\n        :Crime,\n        :Documentary,\n        :Drama,\n        :Fantasy,\n        :FilmNoir,\n        :Horror,\n        :Musical,\n        :Mystery,\n        :Romance,\n        :SciFi,\n        :Thriller,\n        :War,\n        :Western,\n    ],\n)\n\nid2title = Dict()\nfor row in Tables.rows(movie)\n    id2title[row[:movieid]] = row[:movie_title]\nend\n\nrating = rating |> TableOperations.transform(movieid=x->id2title[x])","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"To treat this data as implicit feedback, we replace all the rating by unity:","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"rating = rating |> TableOperations.transform(rating=x->1)","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"Finally, split the dataset to train and test. Several data split methods are implemented in Recommenders, and the below is simple 80/20 split:","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"using Random\nusing Recommenders: ratio_split\nRandom.seed!(1234) # for reproducibility\ntrain_table, test_table = ratio_split(rating, 0.8)","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"Check the first row entry by","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"for row in Tables.rows(rating)\n    print(row)\n    break\nend","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"Tables.ColumnsRow{TableOperations.Transforms{true, TableOperations.Transforms{true, CSV.File, NamedTuple{(:movieid,), Tuple{var\"#1#2\"}}}, NamedTuple{(:rating,), Tuple{var\"#3#4\"}}}}:\n :userid           196\n :movieid             \"Kolya (1996)\"\n :rating             1\n :timestamp  881250949","category":"page"},{"location":"getting_started/#Fit","page":"Getting started","title":"Fit","text":"","category":"section"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"Let's train the recommender model. Here we take matrix factorization model, but the fit API is similar for other models.","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"using Recommenders: ImplicitMF, fit!\n\ndim = 128 # embedding dimension\nuse_bias = true # use bias terms\nreg_coeff = 0.01 # L2 regularization coefficients\n\nmodel = ImplicitMF(dim, use_bias, reg_coeff)\n\nfit!(\n    model,\n    train_table,\n    col_user = :userid, # specify user column\n    col_item = :movieid, # specify item column\n    n_epochs = 3,\n    learning_rate = 0.01,\n    n_negatives = 2, # number of negatives per positive sample\n    verbose=1,\n)","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"By setting verbose=1, one can see the training information:","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"[ Info: epoch=1: train_loss=Inf\n[ Info: epoch=2: train_loss=0.6778364684901991\n[ Info: epoch=3: train_loss=0.5852837966181149","category":"page"},{"location":"getting_started/#Predict","page":"Getting started","title":"Predict","text":"","category":"section"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"Let's get prediction for single user:","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"using Recommenders: predict_u2i\n\nuserid = 10\nn = 3 # number of retrieved items\npred = predict_u2i(\n    model,\n    userid,\n    n,\n    drop_history = true, # whether to drop already consumed items\n)","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"3-element Vector{String}:\n \"Sneakers (1992)\"\n \"Apocalypse Now (1979)\"\n \"Twister (1996)\"","category":"page"},{"location":"getting_started/#Evaluate","page":"Getting started","title":"Evaluate","text":"","category":"section"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"We show how to evaluate the trained model. Let's first make test set aggregated by users","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"user_actioned_items = Dict()\nfor row in Tables.rows(test_table)\n    uid = row[:userid]\n    iid = row[:movieid]\n    if uid in keys(user_actioned_items)\n        push!(user_actioned_items[uid], iid)\n    else\n        user_actioned_items[uid] = [iid]\n    end\nend\ntest_users = collect(keys(user_actioned_items))\nground_truth = collect(values(user_actioned_items))","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"Get predictions for all the users in test set as","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"n = 3\npreds = predict_u2i(\n    model,\n    test_users,\n    n,\n    drop_history = true,\n)","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"Evaluation metrics are implemented as callable struct. For instance, one can evaluate nDCG@10 averaged over all users by","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"using Recommenders: MeanNDCG\nndcg3 = MeanNDCG(3)\nndcg3(preds, ground_truth)","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"0.11568344921472885","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"Note that, in Recommenders.jl, this whole fit → predict → evaluate process is performed by the following evaluate_u2i API:","category":"page"},{"location":"getting_started/","page":"Getting started","title":"Getting started","text":"using Recommenders: evaluate_u2i\n\nmodel = ImplicitMF(dim, use_bias, reg_coeff)\nmetrics = [ndcg3]\nn = 3\n\nevaluate_u2i(\n    model,\n    train_table,\n    test_table,\n    metrics,\n    n,\n    col_user = :userid,\n    col_item = :movieid,\n    n_epochs = 3,\n    learning_rate = 0.01,\n    n_negatives = 2,\n    verbose=1,\n    drop_history = true,\n)","category":"page"},{"location":"models/#Models","page":"Models","title":"Models","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"Pages = [\"models.md\"]","category":"page"},{"location":"models/#Common-interfaces","page":"Models","title":"Common interfaces","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"Modules = [Recommenders]\nOrder   = [:type, :function]\nPages   = [\"base_recommender.jl\"]","category":"page"},{"location":"models/#Recommenders.AbstractRecommender","page":"Models","title":"Recommenders.AbstractRecommender","text":"AbstractRecommender\n\nAbstract struct for all recommendation models.\n\n\n\n\n\n","category":"type"},{"location":"models/#Recommenders.evaluate_u2i-Tuple{AbstractRecommender, Any, Any, Recommenders.MeanMetric, Int64}","page":"Models","title":"Recommenders.evaluate_u2i","text":"evaluate_u2i(model, train_table, test_table, metric, n; kwargs...)\n\nPerform fit! for model on train_table, predict for each user in test_table, and evaluate by metric.\n\nArguments\n\nmodel::AbstractRecommender: model to evaluate.\ntrain_table: any Tables.jl-compatible data for train.\ntest_table: any Tables.jl-compatible data for test.\nmetric: evaluation metrics, MeanMetric or collection of MeanMetric.\nn::Int64: number of retrieved items.\n\nKeyword arguments\n\ndrop_history::Bool: whether to drop already consumed items from predictions.\ncol_user: name of user column in table\ncol_item: name of item column in table\nany model-dependent arguments.\n\nReturn\n\nEvaluated metrics for test_table.\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.fit!-Tuple{AbstractRecommender, Any}","page":"Models","title":"Recommenders.fit!","text":"fit!(model::AbstractRecommender, table; kwargs...)\n\nTrain model by table.\n\nArguments\n\nmodel: concrete type under AbstractRecommender\ntable: any Tables.jl-compatible data for train.\n\nKeyword arguments\n\ncol_user: name of user column in table\ncol_item: name of item column in table\nand other model-dependent arguments.\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.load_model-Tuple{Any}","page":"Models","title":"Recommenders.load_model","text":"load_model(model::AbstractRecommender, filepath)\n\nLoad model by JLD2.\n\nArguments\n\nfilepath: path from which load the model. If the model save multiple files, this argument points to directory.\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.predict_i2i-Tuple{AbstractRecommender, Union{Int64, AbstractString}, Int64}","page":"Models","title":"Recommenders.predict_i2i","text":"predict_i2i(model, itemid, n; kwargs...)\n\nMake recommendations given an item. When itemid is a collection of raw item ids, this function performs parallel predictions by Threads.@threads.\n\nArguments\n\nmodel::AbstractRecommender: trained model.\nitemid:: item id to get predictions. type is AbstractString, Int or their collection.\nn::Int64: number of retrieved items.\n\nKeyword arguments\n\nother model-dependent arguments.\n\nReturn\n\nVector of predicted items, ordered by descending score.\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.predict_u2i-Tuple{AbstractRecommender, Union{Int64, AbstractString}, Int64}","page":"Models","title":"Recommenders.predict_u2i","text":"predict_u2i(model, userid, n; kwargs...)\n\nMake recommendations to user (or users). When userid is a collection of raw user ids, this function performs parallel predictions by Threads.@threads.\n\nArguments\n\nmodel::AbstractRecommender: trained model.\nuserid:: user id to get predictions. type is AbstractString, Int or their collection.\nn::Int64: number of retrieved items.\n\nKeyword arguments\n\ndrop_history::Bool: whether to drop already consumed items from predictions.\nand other model-dependent arguments.\n\nReturn\n\nVector of predicted items, ordered by descending score.\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.save_model","page":"Models","title":"Recommenders.save_model","text":"save_model(model::AbstractRecommender, filepath, overwrite = false)\n\nSave model by JLD2.\n\nArguments\n\nmodel::AbstractRecommender: model to save.\nfilepath: path to save the model. If the model save multiple files, this argument points to directory.\noverwrite: whether to overwrite if filepath already exists.\n\n\n\n\n\n","category":"function"},{"location":"models/#Most-Popular","page":"Models","title":"Most Popular","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"Modules = [Recommenders]\nOrder   = [:type, :function]\nPages   = [\"most_popular.jl\"]","category":"page"},{"location":"models/#Recommenders.MostPopular","page":"Models","title":"Recommenders.MostPopular","text":"MostPopular()\n\nNon-personalized baseline model which predicts n-most popular items in the corpus.\n\n\n\n\n\n","category":"type"},{"location":"models/#Recommenders.fit!-Tuple{MostPopular, Any}","page":"Models","title":"Recommenders.fit!","text":"fit!(model::MostPopular, table; col_user = :userid, col_item = :itemid)\n\nFit most popular model.\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.predict_i2i-Tuple{MostPopular, Union{Int64, AbstractString}, Int64}","page":"Models","title":"Recommenders.predict_i2i","text":"predict_i2i(model::MostPopular, itemid::Union{AbstractString,Int}, n::Int64)\n\nMake n prediction for a give item by most popular model.\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.predict_u2i-Tuple{MostPopular, Union{Int64, AbstractString}, Int64}","page":"Models","title":"Recommenders.predict_u2i","text":"predict_u2i(model::MostPopular, userid::Union{AbstractString,Int}, n::Int64; drop_history::Bool = false)\n\nMake n prediction to user by most popular model.\n\n\n\n\n\n","category":"method"},{"location":"models/#Item-kNN","page":"Models","title":"Item kNN","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"Modules = [Recommenders]\nOrder   = [:type, :function]\nPages   = [\"src/model/item_knn.jl\"]","category":"page"},{"location":"models/#Recommenders.ItemkNN","page":"Models","title":"Recommenders.ItemkNN","text":"ItemkNN(k::Int64, shrink::Float64, weighting::Union{Nothing,Symbol}, weighting_at_inference::Bool, normalize::Bool, normalize_similarity::Bool)\n\nItem-based k-nearest neighborhood algorithm with cosine similarity. The model first computes the item-to-item similarity matrix\n\ns_ij = fracbm r_i cdot bm r_jbm r_ibm r_j + h\n\nwhere r_iu is rating for item i by user u and h is the shrink parameter to suppress the contributions from items with a few ratings.\n\nConstructor arguments\n\nk: size of the nearest neighbors. Only the k-most similar items to each item are stored, which reduces sparse similarity maxrix size, and also make better predictions.\nshrink: shrink paramerer explained above.\nweighting: if set to :ifidf or :bm25, the raw rating matrix is weighted by TF-IDF or BM25, respectively, before computing similarity. If not necessary, just set nothing.\nweighting_at_inference: to use above weighting at inference time, only relevant for BM25.\nnormalize_similarity: if set to true, normalize each column of similarity matrix. See the reference for detail.\n\nReferences\n\nM. Deshpande and G. Karypis (2004), Item-based top-N recommendation algorithms.\n\n\n\n\n\n","category":"type"},{"location":"models/#Recommenders.fit!-Tuple{ItemkNN, Any}","page":"Models","title":"Recommenders.fit!","text":"fit!(model::ItemkNN, table; col_user = :userid, col_item = :itemid, col_rating = :rating)\n\nFit the ItemkNN model. col_rating specifies rating column in the table, which will be all unity if implicit feedback data is given.\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.predict_i2i-Tuple{ItemkNN, Union{Int64, AbstractString}, Int64}","page":"Models","title":"Recommenders.predict_i2i","text":"predict_i2i(model::ItemkNN, itemid::Union{AbstractString,Int}, n::Int64)\n\nMake n prediction for a give item by ItenkNN model.\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.predict_u2i-Tuple{ItemkNN, Union{Int64, AbstractString}, Int64}","page":"Models","title":"Recommenders.predict_u2i","text":"predict_u2i(model::ItemkNN, userid::Union{AbstractString,Int}, n::Int64; drop_history = false)\n\nRecommend top-n items for user by ItemkNN. The predicted rating of item i by user u is computed by\n\n\nhatr_i u = sum_j s_ij r_j u\n\nwhere r_j u is the actual user rating while hatr_i u is the model prediction.\n\n\n\n\n\n","category":"method"},{"location":"models/#Matrix-Factorization","page":"Models","title":"Matrix Factorization","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"Modules = [Recommenders]\nOrder   = [:type, :function]\nPages   = [\"src/model/implicit_mf.jl\"]","category":"page"},{"location":"models/#Recommenders.ImplicitMF","page":"Models","title":"Recommenders.ImplicitMF","text":"ImplicitMF(dim::Int64, use_bias::Bool, reg_coeff::Float64)\n\nMatrix factorization model for implicit feedback. The predicted rating for item i by user u is expreseed as\n\nhat r_ui = mu + b_i + b_u + bm u_u cdot bm v_i\n\nUnlike the model for explicit feedback, the model treats all the (user, item) pairs in the train dataset as positive interaction with label 1, and sample negative (user, item) pairs from the corpus. Currently only the uniform item sampling is implemented. The fitting criteria is the ordinary logloss function\n\n    L = -r_uilog(hat r_ui) - (1 - r_ui)log(1 - hat r_ui)\n\nConstructor arguments\n\ndim: dimension of user/item vectors.\nuse_bias: if set to false, the bias terms (mu, b_i, b_u) are set to zero.\nreg_coeff: L_2 regularization coefficients for model parameters.\n\nReferences\n\nFor instance, Rendle et. al. (2020), Neural Collaborative Filtering vs. Matrix Factorization Revisited .\n\n\n\n\n\n","category":"type"},{"location":"models/#Recommenders.fit!-Tuple{ImplicitMF, Any}","page":"Models","title":"Recommenders.fit!","text":"fit!(model::ImplicitMF, table; callbacks = Any[], col_user = :userid, col_item = :item_id, n_epochs = 2, learning_rate = 0.01, n_negatives = 1, verbose = -1)\n\nFit the ImplicitMF model by stochastic grandient descent (with no batching).\n\nModel-specific arguments\n\nn_epochs: number of epochs. During one epoch, all the row in table is read once.\nlearning_rate: Learing rate of SGD.\nn_negatives: Number of negative item samples per positive (user, item) pair.\nverbose: If set to positive integer, the training info is printed once per verbose.\ncallbacks: Additional callback functions during SGD. One can implement, for instance, monitoring the validation metrics and the early stopping. See Callbacks.\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.predict_u2i-Tuple{ImplicitMF, Union{Int64, AbstractString}, Int64}","page":"Models","title":"Recommenders.predict_u2i","text":"predict_u2i(model::ImplicitMF, userid::Union{AbstractString,Int}, n::Int64; drop_history = false)\n\nMake predictions by using hat r_ui.\n\n\n\n\n\n","category":"method"},{"location":"models/#Bayesian-Personalized-Ranking","page":"Models","title":"Bayesian Personalized Ranking","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"Modules = [Recommenders]\nOrder   = [:type, :function]\nPages   = [\"src/model/bpr.jl\"]","category":"page"},{"location":"models/#Recommenders.BPR","page":"Models","title":"Recommenders.BPR","text":"BPR(dim::Int64, reg_coeff::Float64)\n\nBayesian personalized ranking model. The model evaluates user-item triplet (u i j), which expresses \"the user u prefers item i to item j. Here the following matrix factoriazation model is adopted to model this relation:\n\np_uij = bm u_u cdot bm v_i - bm u_u cdot bm v_j\n\nConstructor arguments\n\ndim: dimension of user/item vectors.\nreg_coeff: L_2 regularization coefficients for model parameters.\n\n\n\n\n\n","category":"type"},{"location":"models/#Recommenders.fit!-Tuple{BPR, Any}","page":"Models","title":"Recommenders.fit!","text":"fit!(model::BPR, table; callbacks = Any[], col_user = :userid, col_item = :item_id, n_epochs = 2, learning_rate = 0.01, n_negatives = 1, verbose = -1)\n\nFit the BPR model by stochastic grandient descent. Instead the learnBPR algorithm proposed by the original paper, the simple SGD with negative sampling is implemented.\n\nModel-specific arguments\n\nn_epochs: number of epochs. During one epoch, all the row in table is read once.\nlearning_rate: Learing rate of SGD.\nn_negatives: Number of negative item samples per positive (user, item) pair.\nverbose: If set to positive integer, the training info is printed once per verbose.\ncallbacks: Additional callback functions during SGD. One can implement, for instance, monitoring the validation metrics and the early stopping. See Callbacks.\n\nReferences\n\nRendel et. al. (2012), BPR: Bayesian Personalized Ranking from Implicit Feedback\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.predict_u2i-Tuple{BPR, Union{Int64, AbstractString}, Int64}","page":"Models","title":"Recommenders.predict_u2i","text":"predict_u2i(model::BPR, userid::Union{AbstractString,Int}, n::Int64; drop_history = false)\n\nMake predictions by using bm u_u cdot bm v_i.\n\n\n\n\n\n","category":"method"},{"location":"models/#Sparse-Linear-Machine","page":"Models","title":"Sparse Linear Machine","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"Modules = [Recommenders]\nOrder   = [:type, :function]\nPages   = [\"src/model/slim.jl\"]","category":"page"},{"location":"models/#Recommenders.SLIM","page":"Models","title":"Recommenders.SLIM","text":"SLIM(l1_ratio::Float64 = 0.5, λminratio::Float64 = 1e-4, k::Int = -1)\n\nSparse linear machine for recommendation, modified with Elastic Net loss. The prediction is made by\n\nhat r_ui = sum_jneq i w_ij r_uj\n\nwhere r_ui is the actual rating for item i by user u, and hat r_ui is the predicted value. w_ij is the model weght matrix. See Refs for algorithm details. SLIM uses Lasso.jl for optimization.\n\nConstructor arguments\n\nl1_ratio: ratio of coefficients between L_1 and L_2 losses. l1_ratio to 0 means the Ridge regularization, while l1_ratio to infty the Lasso.\nλminratio: parameter which governs the strength of regularization. See the docs of Lasso.jl.\nk: the nearest neighborhood size, similar to ItemkNN. If k < 1, the neigoborhood size is infinity.\n\nReferences\n\nX. Ning and G. Karypis (2011), SLIM: Sparse Linear Methods for Top-N Recommender Systems\nM. Levy (2013), Efficient Top-N Recommendation by Linear Regression\n\n\n\n\n\n","category":"type"},{"location":"models/#Recommenders.fit!-Tuple{SLIM, Any}","page":"Models","title":"Recommenders.fit!","text":"fit!(model::SLIM, table; col_user = :userid, col_item = :itemid, col_rating = :rating, cd_tol = 1e-7, nλ = 100)\n\nFit the SLIM model.\n\nModel-specific arguments\n\ncd_tol: tolerance paramerer for convergence, see Lasso.jl\nnλ: length of regularization path, see Lasso.jl\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.predict_i2i-Tuple{SLIM, Union{Int64, AbstractString}, Int64}","page":"Models","title":"Recommenders.predict_i2i","text":"predict_i2i(model::SLIM, itemid::Union{AbstractString,Int}, n::Int64)\n\nMake n prediction for a give item by SLIM model.\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.predict_u2i-Tuple{SLIM, Union{Int64, AbstractString}, Int64}","page":"Models","title":"Recommenders.predict_u2i","text":"predict_u2i(model::SLIM, userid::Union{AbstractString,Int}, n::Int64; drop_history = false)\n\nMake predictions by SLIM model.\n\n\n\n\n\n","category":"method"},{"location":"models/#Random-Walk","page":"Models","title":"Random Walk","text":"","category":"section"},{"location":"models/","page":"Models","title":"Models","text":"Modules = [Recommenders]\nOrder   = [:type, :function]\nPages   = [\"src/model/randomwalk.jl\"]","category":"page"},{"location":"models/#Recommenders.Randomwalk","page":"Models","title":"Recommenders.Randomwalk","text":"Randomwalk()\n\nRecommendation model using random walk with restart on user-item bipartite graph. Implemented algorithm is based on Pixie random walk.\n\nReferences\n\nC.  Eksombatchai (2018), Pixie: A System for Recommending 3+ Billion Items to 200+ Million Users in Real-Time\n\n\n\n\n\n","category":"type"},{"location":"models/#Recommenders.fit!-Tuple{Randomwalk, Any}","page":"Models","title":"Recommenders.fit!","text":"fit!(model::Randomwalk, table; col_user = :userid, col_item = :itemid)\n\nBuild bipartite graph from table. One side of the graph collcets user nodes, and the others item nodes. If a user actions an item, an edge is added between them. The graph is undirected, and has no extra weights.\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.predict_i2i-Tuple{Randomwalk, Union{Int64, AbstractString}, Int64}","page":"Models","title":"Recommenders.predict_i2i","text":"predict_i2i(model::Randomwalk, userid::Union{AbstractString,Int}, n::Int64; drop_history = false, terminate_prob = 0.1, total_walk_length = 10000, min_high_visited_candidates = Inf, high_visited_count_threshold = Inf, pixie_walk_length_scaling = false, aggregate_function = sum)\n\nMake recommendation by random walk with restart. Basic algorithm is as follows:\n\n1. Get users that are connected with the query item by one step. We denote them by ``q \\in Q``.\n2. Starting from each node ``q \\in Q``, perform multiple random walks with certain stop probability. Record the visited count of the items on the walk. We denote the counts of item ``p`` on the walk from ``q`` by ``V_q[p]``.\n3. Finally aggregate ``V_q[p]`` to ``V[p]``, and recommeds top-scored items. Two mothods for aggregation are provided\n- Simple aggregation: Taking sum, ``V[p] = \\sum_{q\\in Q} V_q[p]``. You can also replace `sum` by, for instance, `maximum`.\n- Pixie boosting: ``V[p] = (\\sum_{q\\in Q} \\sqrt{V_q[p]})^2``, putting more importance on the nodes visited by ``q``s.\n\n# Model-specific arguments\n- `terminate_prob`: stop probability of one random walk\n- `total_walk_length`: total walk length over the multiple walks from ``q``'s.\n- `high_visited_count_threshold`: early stopping paramerer. Count up `high_visited_count` when the visited count of certain node reaches this threshold.\n- `min_high_visited_candidates`: early stopping parameter. Terminate the walk from some node ``q`` if `high_visited_count` reaches `min_high_visited_candidates`.\n- `pixie_walk_length_scaling`: If set to true, the start node ``q`` with more degree will be given more walk length. If false, the walk length is the same over all the nodes ``q \\in Q``\n- `pixie_multi_hit_boosting`: If true, pixie boosting is adopted for aggregation. If false, simple aggregation is used.\n- `aggregate_function`: function used by simple aggregation.\n\n\n\n\n\n","category":"method"},{"location":"models/#Recommenders.predict_u2i-Tuple{Randomwalk, Union{Int64, AbstractString}, Int64}","page":"Models","title":"Recommenders.predict_u2i","text":"predict_u2i(model::Randomwalk, userid::Union{AbstractString,Int}, n::Int64; drop_history = false, terminate_prob = 0.1, total_walk_length = 10000, min_high_visited_candidates = Inf, high_visited_count_threshold = Inf, pixie_walk_length_scaling = false, pixie_multi_hit_boosting = false, aggregate_function = sum)\n\nMake recommendation by random walk with restart. Basic algorithm is as follows:\n\nGet items that are already consumed by the user (on the graph, they are connected by one step). We denote them by q in Q.\nStarting from each node q in Q, perform multiple random walks with certain stop probability. Record the visited count of the items on the walk. We denote the counts of item p on the walk from q by V_qp.\nFinally aggregate V_qp to Vp, and recommeds top-scored items. Two mothods for aggregation are provided\n\nSimple aggregation: Taking sum, Vp = sum_qin Q V_qp. You can also replace sum by, for instance, maximum.\nPixie boosting: Vp = (sum_qin Q sqrtV_qp)^2, putting more importance on the nodes visited by qs.\n\nModel-specific arguments\n\nterminate_prob: stop probability of one random walk\ntotal_walk_length: total walk length over the multiple walks from q's.\nhigh_visited_count_threshold: early stopping paramerer. Count up high_visited_count when the visited count of certain node reaches this threshold.\nmin_high_visited_candidates: early stopping parameter. Terminate the walk from some node q if high_visited_count reaches min_high_visited_candidates.\npixie_walk_length_scaling: If set to true, the start node q with more degree will be given more walk length. If false, the walk length is the same over all the nodes q in Q\npixie_multi_hit_boosting: If true, pixie boosting is adopted for aggregation. If false, simple aggregation is used.\naggregate_function: function used by simple aggregation.\n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Recommenders","category":"page"},{"location":"#Recommenders","page":"Home","title":"Recommenders","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package aims to provide light-weight recommendation models, mainly for implicit feedback data. We want to provide","category":"page"},{"location":"","page":"Home","title":"Home","text":"consistent interface for model training and inference\nflexibility for input data with Tables.jl package, which offers simple, but powerful abstract interface for tabular data\nrobust baseline metrics for classic datasets. The comparison of advanced recommendation models to these baselines turns out to be challenge [1, 2].","category":"page"},{"location":"","page":"Home","title":"Home","text":"See Getting started for quick start. More advanced usage is scripted in examples.","category":"page"},{"location":"","page":"Home","title":"Home","text":"[1]: M. F. Dacrema et. al., Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches","category":"page"},{"location":"","page":"Home","title":"Home","text":"[2]: S. Rendle, Evaluation Metrics for Item Recommendation under Sampling","category":"page"},{"location":"utils/#Utilities","page":"Utilities","title":"Utilities","text":"","category":"section"},{"location":"utils/#Data-utilities","page":"Utilities","title":"Data utilities","text":"","category":"section"},{"location":"utils/","page":"Utilities","title":"Utilities","text":"Modules = [Recommenders]\nOrder   = [:type, :function]\nPages   = [\"src/dataset/data_split.jl\"]","category":"page"},{"location":"utils/#Recommenders.leave_one_out_split-Tuple{Any}","page":"Utilities","title":"Recommenders.leave_one_out_split","text":"leave_one_out_split(table; col_user = :userid, col_time = :timestamp)\n\nLeave-one-out split for the input table. For each user, whose column is specifed by col_user, the items are sorted by col_time, and the last one is split into the test set. The others remain in the train set.\n\nReturns\n\ntrain_table\ntest_table\n\n\n\n\n\n","category":"method"},{"location":"utils/#Recommenders.ratio_split","page":"Utilities","title":"Recommenders.ratio_split","text":"ratio_split(table, train_ratio = 0.7)\n\nSplit the table randomly, with the train set ratio specifed by train_ratio argument. Current implementaion assumes table object that can be converted to DataFrame.\n\nReturns\n\ntrain_table\ntest_table\n\n\n\n\n\n","category":"function"},{"location":"utils/#Callbacks","page":"Utilities","title":"Callbacks","text":"","category":"section"},{"location":"utils/","page":"Utilities","title":"Utilities","text":"For models trained by Stochastic Gradient Descent (SGD), one can give callback functions to the fit! method. Callback is any callable (functions or callable structs) that takes model, train_loss, epoch and verbose as inputs. If StopTrain exception is raised by the callback, training loop stops before the completion.","category":"page"},{"location":"utils/","page":"Utilities","title":"Utilities","text":"Currently only the following callbacks are implemented","category":"page"},{"location":"utils/","page":"Utilities","title":"Utilities","text":"Modules = [Recommenders]\nOrder   = [:type, :function]\nPages   = [\"src/model/utils.jl\"]","category":"page"},{"location":"utils/#Recommenders.EvaluateValidData","page":"Utilities","title":"Recommenders.EvaluateValidData","text":"EvaluateValidData(valid_metric::MeanMetric, valid_table, early_stopping_rounds, name = \"val_metric\")\n\nCallback to monitor the validation metrics during training, and raise StopTrain exception if early stopping is requred.\n\nConstructor arguments\n\nvalid_metric: monotring metric. See Evaluation metrics for the available ones.\nvalid_table: any Tables.jl-compatible object for validation dataset.\nearly_stopping_rounds: If the validation metric does not improve more than this epochs, the early stopping is invoked. If set to be less than 1, no early stopping is applied.\nname: metrics name to show on logger.\n\nExample\n\nUse in the matrix factorizaion training.\n\nndcg10 = MeanNDCG(10)\ncb = EvaluateValidData(ndcg10, test_table, 1, \"val_NDCG\")\n\nmodel = ImplicitMF(16, true, 0.01)\nfit!(\n    model,\n    train_table,\n    10,\n    callbacks = [cb],\n    col_item = :movieid,\n    n_epochs = 20,\n    n_negatives = 1,\n    learning_rate = 0.01,\n    verbose = 1,\n)\n\n\n\n\n\n","category":"type"}]
}