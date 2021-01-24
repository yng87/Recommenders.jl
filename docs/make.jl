using Recommender
using Documenter

makedocs(;
    modules=[Recommender],
    authors="yng87 <k.yanagi07@gmail.com> and contributors",
    repo="https://github.com/yng87/Recommender.jl/blob/{commit}{path}#L{line}",
    sitename="Recommender.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://yng87.github.io/Recommender.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/yng87/Recommender.jl",
)
