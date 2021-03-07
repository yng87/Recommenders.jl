abstract type Movielens <: AbstractDataset end

function load(d::Movielens)
    # Movielens dataset has different kinds of meta data,
    # depending on the size of rating.
    # So we leave load function as abstract.
    error("load method is not implemented.")
end

function load_rating(d::Movielens)
    error("load_rating method is not implemented.")
end

function download(d::Movielens; kwargs...)
    download_zip(d.url, d.name, d.dataset_dir; kwargs...)
    return
end

"""
Movielens100k
"""
@with_kw_noshow struct Movielens100k <: Movielens
    name::AbstractString = "movielens100k"
    url::AbstractString = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
    dataset_dir::AbstractString = joinpath(@__DIR__, "..", "..", "dataset", "movielens100k")
end

function load(d::Movielens100k)
    return load_rating(d), load_user(d), load_item(d)
end

function load_rating(d::Movielens100k)
    df = loadcsv(
        joinpath(d.dataset_dir, "u.data"),
        delim = "\t",
        header = 0,
        columns = [:userid, :movieid, :rating, :timestamp],
    )
    return df
end

function load_user(d::Movielens100k)
    df = loadcsv(
        joinpath(d.dataset_dir, "u.user"),
        delim = "|",
        header = 0,
        columns = [:userid, :age, :gender, :occupation, :zipcode],
    )
    return df
end

function load_item(d::Movielens100k)
    df = loadcsv(
        joinpath(d.dataset_dir, "u.item"),
        delim = "|",
        header = 0,
        columns = [
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
        ],
    )
    return df
end

"""
Movielens1M
"""

@with_kw_noshow struct Movielens1M <: Movielens
    name::AbstractString = "movielens1m"
    url::AbstractString = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    dataset_dir::AbstractString = joinpath(@__DIR__, "..", "..", "dataset", "movielens1m")
end

function load(d::Movielens1M)
    return load_rating(d), load_user(d), load_item(d)
end

function load_rating(d::Movielens1M)
    df = loadcsv(
        joinpath(d.dataset_dir, "ratings.dat"),
        delim = "::",
        header = 0,
        columns = [:userid, :movieid, :rating, :timestamp],
    )
    return df
end

function load_user(d::Movielens1M)
    df = loadcsv(
        joinpath(d.dataset_dir, "users.dat"),
        delim = "::",
        header = 0,
        columns = [:userid, :gender, :age, :occupation, :zipcode],
    )
    return df
end

function load_item(d::Movielens1M)
    df = loadcsv(
        joinpath(d.dataset_dir, "movies.dat"),
        delim = "::",
        header = 0,
        columns = [:movieid, :title, :genres],
    )
    return df
end