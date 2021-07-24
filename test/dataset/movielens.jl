using Tables
using Recommender: Movielens1M, Movielens100k, download, load_dataset
using Test

@testset "Movielens100k" begin
    ml100k = Movielens100k()
    download(ml100k, usecache = false, removezip = true, unzip = true)
    rating, user, movie = load_dataset(ml100k)

    @test size(rating) == (100000,)
    @test Tables.columnnames(rating) == [:userid, :movieid, :rating, :timestamp]

    @test size(user) == (943,)
    @test Tables.columnnames(user) == [:userid, :age, :gender, :occupation, :zipcode]

    @test size(movie) == (1682,)
    @test Tables.columnnames(movie) == [
        :movieid,
        :movie_title,
        :release_date,
        :video_release_date,
        :IMDbURL,
        :unknown,
        :Action,
        :Adventure,
        :Animation,
        :Childrens,
        :Comedy,
        :Crime,
        :Documentary,
        :Drama,
        :Fantasy,
        :FilmNoir,
        :Horror,
        :Musical,
        :Mystery,
        :Romance,
        :SciFi,
        :Thriller,
        :War,
        :Western,
    ]
end

@testset "Movielent1M" begin
    ml1m = Movielens1M()
    download(ml1m, usecache = false, removezip = true, unzip = true)
    rating, user, movie = load_dataset(ml1m)

    @test size(rating) == (1000209,)
    @test Tables.columnnames(rating) == [:userid, :movieid, :rating, :timestamp]

    @test size(user) == (6040,)
    @test Tables.columnnames(user) == [:userid, :gender, :age, :occupation, :zipcode]
    @test min(user.userid...) == 1
    @test max(user.userid...) == 6040

    @test size(movie) == (3883,)
    @test Tables.columnnames(movie) == [:movieid, :title, :genres]
    @test min(movie.movieid...) == 1
    @test max(movie.movieid...) == 3952
end
