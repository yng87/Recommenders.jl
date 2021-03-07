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

function loadcsv(d::Movielens, filename::AbstractString, columns)
    df = loadcsv(
        joinpath(d.dataset_dir, filename),
        delim = d.delim,
        header = d.header,
        columns = columns,
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
    delim::AbstractString = "::"
    header::Int = 0
end

function load(d::Movielens1M)
    return load_rating(d), load_user(d), load_item(d)
end

function load_rating(d::Movielens1M)
    return loadcsv(d, "ratings.dat", [:userid, :movieid, :rating, :timestamp])
end

function load_user(d::Movielens1M)
    return loadcsv(d, "users.dat", [:userid, :gender, :age, :occupation, :zipcode])
end

function load_item(d::Movielens1M)
    return loadcsv(d, "movies.dat", [:movieid, :title, :genres])
end