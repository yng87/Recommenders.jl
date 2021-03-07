function loadcsv(path::AbstractString; delim = ",", header = 0, columns = nothing)
    df = DataFrame(CSV.File(path, delim = delim, header = header))
    if !isnothing(columns)
        rename!(df, columns)
    end
    return df
end