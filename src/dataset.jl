global __datasets = nothing

abstract type AbstractDataset end

struct Dataset <: AbstractDataset
    name::AbstractString
    format::AbstractString
    url::AbstractString
    dirpath::AbstractString
end

Dataset(name, format, url) = Dataset(name, format, url, joinpath(@__DIR__, "..", "dataset", name))

function datasets()
    if __datasets === nothing
        path = joinpath(@__DIR__, "..", "dataset", "datasets.csv")
        global __datasets = DataFrame(CSV.File(path))
    end
    return __datasets::DataFrame
end

function datasets(name::AbstractString)
    df = datasets()
    return df[findall(isequal(name), df[:, :name]), :]
end

function dataset(name::AbstractString)
    df = datasets(name)
    if isempty(df)
        error("$name not exist in our dataset list.")
    end
    # get only first element
    name = df[1, :name]
    format = df[1, :format]
    url = df[1, :url]

    return Dataset(name, format, url)
end

function download(dataset::AbstractDataset)
    error("download method is not implemented.")
end

function load(dataset::AbstractDataset)
    error("load method is not implemented.")
end

function download(dataset::Dataset; kwargs...)
    if dataset.format == "zip"
        download_zip(dataset.url, dataset.name, dataset.dirpath; kwargs...)
        return
    end
end

function load(dataset::Dataset; kwargs...)
    if dataset.name == "movielens1m"
        return load_movielens1m(dataset.dirpath)
    end
end