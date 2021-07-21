abstract type Movielens <: AbstractDataset end

function load_dataset(d::Movielens)
    # Movielens dataset has different kinds of meta data,
    # depending on the size of rating.
    # So we leave load function as abstract.
    error("load method is not implemented.")
end

function load_inter(d::Movielens)
    error("load_inter method is not implemented.")
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

function load_dataset(d::Movielens100k)
    return load_inter(d), load_user(d), load_item(d)
end

function load_inter(d::Movielens100k)
    csv = CSV.File(
        joinpath(d.dataset_dir, "u.data"),
        delim = "\t",
        header = [:userid, :movieid, :rating, :timestamp],
    )
    return csv
end

function load_user(d::Movielens100k)
    csv = CSV.File(
        joinpath(d.dataset_dir, "u.user"),
        delim = "|",
        header = [:userid, :age, :gender, :occupation, :zipcode],
    )
    return csv
end

function load_item(d::Movielens100k)
    csv = CSV.File(
        joinpath(d.dataset_dir, "u.item"),
        delim = "|",
        header = [
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
    return csv
end

"""
Movielens1M
"""

@with_kw_noshow struct Movielens1M <: Movielens
    name::AbstractString = "movielens1m"
    url::AbstractString = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"
    dataset_dir::AbstractString = joinpath(@__DIR__, "..", "..", "dataset", "movielens1m")
end

function load_dataset(d::Movielens1M)
    return load_inter(d), load_user(d), load_item(d)
end

function load_inter(d::Movielens1M)
    csv = CSV.File(
        joinpath(d.dataset_dir, "ratings.dat"),
        delim = "::",
        header = [:userid, :movieid, :rating, :timestamp],
    )
    return csv
end

function load_user(d::Movielens1M)
    csv = CSV.File(
        joinpath(d.dataset_dir, "users.dat"),
        delim = "::",
        header = [:userid, :gender, :age, :occupation, :zipcode],
    )
    return csv
end

function load_item(d::Movielens1M)
    csv = CSV.File(
        joinpath(d.dataset_dir, "movies.dat"),
        delim = "::",
        header = [:movieid, :title, :genres],
    )
    return csv
end