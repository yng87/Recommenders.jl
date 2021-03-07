using Recommender: Movielens1M, Movielens100k
using Test

println("test Movielens1M")

ml1m = Movielens1M()
download(ml1m, usecache = false, removezip = true, unzip = true)
rating, user, movie = load(ml1m)

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

println("test Movielens100k")

ml100k = Movielens100k()
download(ml100k, usecache = false, removezip = true, unzip = true)
rating, user, movie = load(ml100k)

@test size(rating) == (100000, 4)
@test names(rating) == ["userid", "movieid", "rating", "timestamp"]

@test size(user) == (943, 5)
@test names(user) == ["userid", "age", "gender", "occupation", "zipcode"]

@test size(movie) == (1682, 24)
@test names(movie) == [
    "movieid",
    "movie_title",
    "release_date",
    "video_release_date",
    "IMDbURL",
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Childrens",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "FilmNoir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "SciFi",
    "Thriller",
    "War",
    "Western",
]