function loadcsv(path::AbstractString; delim = ",", header = 0, columns = nothing)
    df = DataFrame(CSV.File(path, delim = delim, header = header))
    if !isnothing(columns)
        rename!(df, columns)
    end
    return df
end

function load_movielens1m(dirpath::AbstractString)
    rating = loadcsv(
        joinpath(dirpath, "ratings.dat"),
        delim = "::",
        header = 0,
        columns = [:userid, :movieid, :rating, :timestamp],
    )
    user = loadcsv(
        joinpath(dirpath, "users.dat"),
        delim = "::",
        header = 0,
        columns = [:userid, :gender, :age, :occupation, :zipcode],
    )
    movie = loadcsv(
        joinpath(dirpath, "movies.dat"),
        delim = "::",
        header = 0,
        columns = [:movieid, :title, :genres],
    )
    return rating, user, movie
end