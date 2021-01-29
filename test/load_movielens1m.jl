using Recommender
using Test

d = dataset("movielens1m")
download(d, usecache=false, removezip=true, unzip=true)
rating, user, movie = load(d)

@test size(rating) == (1000209, 4)
@test names(rating) == ["userid", "movieid", "rating", "timestamp"]

@test size(user) == (6040, 5)
@test names(user) == ["userid", "gender", "age", "occupation", "zipcode"]
@test min(user.userid...) == 1
@test max(user.userid...) == 6040

@test size(movie) == (3883, 3)
@test names(movie) == ["movieid", "title", "genres"]
@test min(movie.movieid...) == 1
@test max(movie.movieid...) == 3952
