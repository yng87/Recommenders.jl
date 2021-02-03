const OPERATIONS = (:predict_i2i, :predict_u2i)

function predict_i2i(m::MLJBase.Model, fitresult, itemids; kwargs...)
    error("predict_i2i is not not implemented for recommender type $(typeof(m))")
end

function predict_u2i(m::MLJBase.Model, fitresult, userids; kwargs...)
    error("predict_u2i is not not implemented for recommender type $(typeof(m))")
end

for operation in OPERATIONS
    # similar code used in MLJBase
    ex = quote
        function $operation(mach::Machine, Xraw)
            if mach.state > 0
                return $(operation)(mach.model,
                                    mach.fitresult,
                                    reformat(mach.model, Xraw)...)
            else
                error("$mach has not been trained.")
            end
        end
    end
    eval(ex)
end