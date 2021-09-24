# Utilities

## Data utilities
```@autodocs
Modules = [Recommenders]
Order   = [:type, :function]
Pages   = ["src/dataset/data_split.jl"]
```

## Callbacks

For models trained by Stochastic Gradient Descent (SGD), one can give callback functions to the `fit!` method. Callback is any callable (functions or callable structs) that takes `model`, `train_loss`, `epoch` and `verbose` as inputs. If `StopTrain` exception is raised by the callback, training loop stops before the completion.

Currently only the following callbacks are implemented
```@autodocs
Modules = [Recommenders]
Order   = [:type, :function]
Pages   = ["src/model/utils.jl"]
```