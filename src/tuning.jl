"""
    recommended_ranges(model)

NamedTuple of suggested MLJ search specifications for a TDA model. Numeric
entries contain `lower`, `upper`, and optionally `scale`; categorical entries
contain `values`. These are starting points, not universal optimal values.
Bandwidths and cutoffs assume coordinates/distances on a scale around one.
Adapt them to physical units, training sample size, and computation budget.
Use [`tuning_ranges`](@ref) to construct MLJ ranges for a complete pipeline.
"""
recommended_ranges(::MMI.Model) = NamedTuple()
recommended_ranges(::PersistenceImageVectorizer) = (
    sigma=(lower=0.01,upper=1.0,scale=:log),
    slope_end=(lower=0.1,upper=1.0),
    width=(lower=5,upper=20),height=(lower=5,upper=20))
recommended_ranges(::PersistenceLandscapeVectorizer) = (
    n_landscapes=(lower=1,upper=5),length=(lower=10,upper=100))
recommended_ranges(::PersistenceCurveVectorizer) = (
    length=(lower=10,upper=100),integrate=(values=[true,false],))
recommended_ranges(::PersistenceDescriptorVectorizer) = (
    resolution=(lower=5,upper=100),size=(values=[(3,3),(5,5),(10,10)],),
    order=(lower=1,upper=10),n_centers=(lower=2,upper=20),
    alpha=(lower=0.25,upper=2.0),birth_cap=(lower=0.5,upper=5.0))

function _range_specs(model::PersistenceDescriptorVectorizer)
    specs = recommended_ranges(model)
    fields = if model.kind in (:euler,:polynomial,:topological)
        (:resolution,)
    elseif model.kind==:block
        (:size,:alpha)
    elseif model.kind==:tent
        (:size,)
    elseif model.kind==:tropical
        (:order,:birth_cap)
    else
        (:n_centers,)
    end
    return (; (field=>getproperty(specs,field) for field in fields)...)
end
_range_specs(model) = recommended_ranges(model)

_nested_field(prefix,field) = prefix===nothing ? field : Expr(:.,prefix,QuoteNode(field))

"""
    tuning_ranges(model; recursive=true)

Construct MLJ `range` objects from [`recommended_ranges`](@ref), recursively
including supported models nested in a PH model or `Pipeline`. Load MLJ or
MLJBase before calling. Heavy MLJ dependencies are not required for native TDA.
Only relevant hyperparameters of `PersistenceDescriptorVectorizer.kind` are
included. Ranges of unrelated classifier models remain the caller's choice.
Pass this result directly to `TunedModel(..., range=tuning_ranges(pipeline))`.
Cross-validation must wrap the entire PH/vectorization/classifier pipeline so
grids, bandwidths learned from data, and codebooks are fitted within each fold.
"""
function tuning_ranges(model::MMI.Model;recursive::Bool=true)
    ranges = Any[]
    function visit(part,prefix)
        for (field,spec) in pairs(_range_specs(part))
            path = _nested_field(prefix,field)
            applicable(Base.range,model,path) ||
                throw(ArgumentError("load MLJ or MLJBase before calling tuning_ranges"))
            push!(ranges,Base.range(model,path;spec...))
        end
        if recursive
            for field in fieldnames(typeof(part))
                child = getfield(part,field)
                child isa MMI.Model && visit(child,_nested_field(prefix,field))
            end
        end
    end
    visit(model,nothing)
    return ranges
end

for T in (PersistenceImageVectorizer,PersistenceLandscapeVectorizer,
        PersistenceCurveVectorizer,PersistenceDescriptorVectorizer)
    @eval MMI.hyperparameter_ranges(::Type{$T}) =
        Tuple(get(recommended_ranges($T()),field,nothing) for field in fieldnames($T))
end
