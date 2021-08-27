using Recommenders
using Documenter

makedocs(;
    modules = [Recommenders],
    authors = "yng87 <k.yanagi07@gmail.com> and contributors",
    repo = "https://github.com/yng87/Recommenders.jl/blob/{commit}{path}#L{line}",
    sitename = "Recommenders.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://yng87.github.io/Recommenders.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/yng87/Recommenders.jl")
