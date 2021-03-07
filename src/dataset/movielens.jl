abstract type Movielens <: AbstractDataset end

function load(d::Movielens)
    error("load method is not implemented.")
end

function load_rating(d::Movielens)
    error("load_rating method is not implemented.")
end

function load_user(d::Movielens)
    error("load_user method is not implemented.")
end

function load_item(d::Movielens)
    error("load_item method is not implemented.")
end

function download(d::Movielens)
    error("download method is not implemented.")
end

"""
Movielens1M
"""

@with_kw_noshow struct Movielens1M <: Movielens
    url::AbstractString = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    dataset_dir::AbstractString = joinpath(@__DIR__, "..", "..", "dataset", "movielens1m")
    delim::AbstractString = "::"
    header::Int = 0
end

function load(d::Movielens1M)
    return load_rating(d), load_user(d), load_item(d)
end

function load_rating(d::Movielens1M)
    rating = loadcsv(
        joinpath(d.dataset_dir, "ratings.dat"),
        delim = d.delim,
        header = d.header,
        columns = [:userid, :movieid, :rating, :timestamp],
    )
    return rating
end

function load_user(d::Movielens1M)
    user = loadcsv(
        joinpath(d.dataset_dir, "users.dat"),
        delim = d.delim,
        header = d.header,
        columns = [:userid, :gender, :age, :occupation, :zipcode],
    )
    return user
end

function load_item(d::Movielens1M)
    movie = loadcsv(
        joinpath(d.dataset_dir, "movies.dat"),
        delim = d.delim,
        header = d.header,
        columns = [:movieid, :title, :genres],
    )
    return movie
end

function download(d::Movielens1M; kwargs...)
    download_zip(d.url, "movielens1m", d.dataset_dir; kwargs...)
    return
end