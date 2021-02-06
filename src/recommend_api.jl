const OPERATIONS = (:retrieve, :ranksort)

function retrieve(m::MLJBase.Model, fitresult, itemids; kwargs...)
    error("retrieve is not not implemented for recommender type $(typeof(m))")
end

function ranksort(m::MLJBase.Model, fitresult, userids; kwargs...)
    error("ranksort not not implemented for recommender type $(typeof(m))")
end

for operation in OPERATIONS
    # similar code used in MLJBase
    ex = quote
        function $operation(mach::Machine, X, k)
            if mach.state > 0
                return $(operation)(mach.model,
                                    mach.fitresult,
                                    X,
                                    k)
            else
                error("$mach has not been trained.")
            end
        end
    end
    eval(ex)
end