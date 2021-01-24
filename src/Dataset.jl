using HTTP, ZipFile

abstract type Dataset end

function download(dataset::Dataset)
    error("download method is not implemented.")
end

struct ZipDataset <: Dataset
    dirname::AbstractString
    url::AbstractString
end

function download(dataset::ZipDataset; usecache=true, unzip=true, removezip=false)
    dirpath = joinpath(@__DIR__, "..", "cache", "dataset", dataset.dirname)

    if usecache && isdir(dirpath)
        return
    end
    if isdir(dirpath)
        rm(dirpath, force=true, recursive=true)
    end

    # download zip
    mkdir(dirpath)
    zippath = joinpath(dirpath, "$(dataset.dirname).zip")
    io = open(zippath, "w")
    HTTP.request("GET", dataset.url, response_stream=io)

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

# TODO: move info to csv file or so.
ml_1m = ZipDataset("ml-1m", "http://files.grouplens.org/datasets/movielens/ml-1m.zip")
download(ml_1m, usecache=false)