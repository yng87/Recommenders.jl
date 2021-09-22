# Models

```@contents
Pages = ["models.md"]
```

## Common interfaces

```@autodocs
Modules = [Recommenders]
Order   = [:type, :function]
Pages   = ["base_recommender.jl"]
```

## Most Popular
```@autodocs
Modules = [Recommenders]
Order   = [:type, :function]
Pages   = ["most_popular.jl"]
```

## Item kNN
```@autodocs
Modules = [Recommenders]
Order   = [:type, :function]
Pages   = ["src/model/item_knn.jl"]
```

## Matrix Factorization
```@autodocs
Modules = [Recommenders]
Order   = [:type, :function]
Pages   = ["src/model/implicit_mf.jl"]
```

## Bayesian Personalized Ranking
```@autodocs
Modules = [Recommenders]
Order   = [:type, :function]
Pages   = ["src/model/bpr.jl"]
```

## Sparse Linear Machine
```@autodocs
Modules = [Recommenders]
Order   = [:type, :function]
Pages   = ["src/model/slim.jl"]
```

## Random Walk
```@autodocs
Modules = [Recommenders]
Order   = [:type, :function]
Pages   = ["src/model/randomwalk.jl"]
```