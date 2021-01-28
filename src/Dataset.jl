global __datasets = nothing

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

# dataset struct

abstract type AbstractDataset end

function download(dataset::AbstractDataset)
    error("download method is not implemented.")
end

function load(dataset::AbstractDataset)
    error("load method is not implemented.")
end

struct Dataset <: AbstractDataset
    name::AbstractString
    format::AbstractString
    url::AbstractString
    dirpath::AbstractString
end

Dataset(name, format, url) = Dataset(name, format, url, joinpath(@__DIR__, "..", "dataset", name))

function load(dataset::Dataset; kwargs...)
    if dataset.name == "movielens1m"
        return load_movielens1m(dataset.dirpath)
    end
end

function loadcsv(path::AbstractString; delim=",", header=0, columns=nothing)
    df = DataFrame(CSV.File(path, delim=delim, header=header))
    return rename!(df, columns)
end

function load_movielens1m(dirpath::AbstractString)
    rating = loadcsv(joinpath(dirpath, "ratings.dat"), delim="::", header=0,
        columns=[:userid, :movieid, :rating, :timestamp])
    user = loadcsv(joinpath(dirpath, "users.dat"), delim="::", header=0,
        columns=[:userid, :gender, :age, :occupation, :zipcode])
    movie = loadcsv(joinpath(dirpath, "movies.dat"), delim="::", header=0,
        columns=[:movieid, :title, :genres])
    return rating, user, movie
end

function download(dataset::Dataset; kwargs...)
    if dataset.format == "zip"
        download_zip(dataset.url, dataset.name, dataset.dirpath; kwargs...)
        return
    end
end

function download_zip(url::AbstractString, name::AbstractString, dirpath::String; usecache=true, unzip=true, removezip=false)
    if usecache && isdir(dirpath)
        return
    end
    if isdir(dirpath)
        rm(dirpath, force=true, recursive=true)
    end

    # download zip
    mkdir(dirpath)
    zippath = joinpath(dirpath, "$(name).zip")
    
    try
        io = open(zippath, "w")
        HTTP.request("GET", url, response_stream=io)
    catch e
        isfile(zippath)
        rm(zippath)
        throw(e)
    end
   

    # unzip
    if unzip
        r = ZipFile.Reader(zippath)
        for f in r.files
            if f.name[end]=='/'
                continue
            end
            println("Filename: $(f.name)")
            io = open(joinpath(dirpath, basename(f.name)), "w")
            write(io, read(f, String))
        end
        close(r)
    end

    # remove zip
    if removezip
        rm(zippath)
    end
end