# from https://github.com/JuliaAI/MLJBase.jl/blob/dev/src/operations.jl
_err_serialized(operation) = throw(
    ArgumentError(
        "Calling $operation on a " * "deserialized machine with no data " * "bound to it. ",
    ),
)

function retrieve(mach::MLJBase.Machine{<:MLJBase.Model,true}, x, n)
    isempty(mach.args) && _err_serialized(:retrieve)
    model = mach.model
    return retrieve(model, mach.fitresult, x, n)
end

# const OPERATIONS = (:retrieve,)

# for operation in OPERATIONS

#     ex = quote
#         function $(operation)(mach::Machine{<:Model,false}; rows = :)
#             # catch deserialized machine with no data:
#             isempty(mach.args) && _err_serialized($operation)
#             return ($operation)(mach, mach.args[1](rows = rows))
#         end
#         function $(operation)(mach::Machine{<:Model,true}; rows = :)
#             # catch deserialized machine with no data:
#             isempty(mach.args) && _err_serialized($operation)
#             model = mach.model
#             return ($operation)(
#                 model,
#                 mach.fitresult,
#                 selectrows(model, rows, mach.data[1])...,
#             )
#         end
#     end
#     eval(ex)

# end
