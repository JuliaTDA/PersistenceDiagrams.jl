"""
    PersistenceDescriptorVectorizer(; kind=:tent, resolution=20, size=(5,5),
        order=5, n_centers=8, seed=42, alpha=1.0, birth_cap=1.0, polynomial_type=:S)

MLJ transformer for `:euler`, `:block`, `:tropical`, `:tent`, `:polynomial`,
`:topological`, or `:atol` descriptors. Every grid/codebook is fitted exclusively
on training diagrams. Ordinary descriptors concatenate features across diagram
columns. `:euler` combines all homology dimensions in a row into one ECC and
requires dimension metadata in each diagram. Complex coefficients are represented
by real/imaginary columns. Empty training dimensions produce fixed-size output.
"""
MMI.@mlj_model mutable struct PersistenceDescriptorVectorizer <: AbstractVectorizer
    kind::Symbol = :tent::(_ in (:euler,:block,:tropical,:tent,:polynomial,:topological,:atol))
    resolution::Int = 20::(_ >= 2)
    size::Tuple{Int,Int} = (5,5)::(all(>(0),_))
    order::Int = 5::(_ > 0)
    n_centers::Int = 8::(_ > 0)
    seed::Int = 42
    alpha::Float64 = 1.0::(0 < _ <= 2)
    birth_cap::Float64 = 1.0::(_ > 0)
    polynomial_type::Symbol = :S::(_ in (:R,:S,:T))
end

function vectorizer(model::PersistenceDescriptorVectorizer,diagrams)
    if model.kind==:block
        return PersistenceBlock(diagrams;size=model.size,alpha=model.alpha)
    elseif model.kind==:tent
        return TentTemplate(diagrams;size=model.size)
    elseif model.kind==:tropical
        return TropicalCoordinates(;order=model.order,birth_cap=model.birth_cap)
    elseif model.kind==:polynomial
        return ComplexPolynomial(;length=model.resolution,polynomial_type=model.polynomial_type)
    elseif model.kind==:topological
        return TopologicalVector(;length=model.resolution)
    elseif model.kind==:atol
        return Atol(diagrams;n_centers=model.n_centers,rng=MersenneTwister(model.seed))
    else
        points = reduce(vcat,_finite_points.(diagrams);init=Tuple{Float64,Float64}[])
        if isempty(points)
            lo,hi = 0.0,1.0
        else
            lo = minimum(first,points)
            hi = maximum(last,points)
            hi = max(hi,lo+eps(max(1.0,abs(lo))))
        end
        return EulerCharacteristicCurve(range(lo,hi;length=model.resolution))
    end
end

function MMI.fit(model::PersistenceDescriptorVectorizer,verbosity::Int,X)
    if model.kind!=:euler
        fitted = map(Tables.columnnames(X)) do col
            col=>vectorizer(model,vec(Tables.getcolumn(X,col)))
        end
        return fitted,nothing,NamedTuple()
    end
    columns = Tables.columnnames(X)
    diagrams = [d for col in columns for d in Tables.getcolumn(X,col)]
    all(d -> get(d.meta,:dim,missing) isa Integer,diagrams) ||
        throw(ArgumentError("Euler vectorization requires diagram dimension metadata"))
    fitted = (columns=columns,curve=vectorizer(model,diagrams))
    return fitted,nothing,NamedTuple()
end

function MMI.transform(model::PersistenceDescriptorVectorizer,fitted,X)
    if model.kind!=:euler
        return invoke(MMI.transform,Tuple{AbstractVectorizer,Any,Any},model,fitted,X)
    end
    matrix = zeros(MMI.nrows(X),output_size(fitted.curve))
    for (i,row) in enumerate(Tables.rows(X))
        diagrams = [Tables.getcolumn(row,col) for col in fitted.columns]
        matrix[i,:] .= fitted.curve(diagrams)
    end
    return MMI.table(matrix;names=[Symbol(:euler_,j) for j in axes(matrix,2)])
end

MMI.metadata_pkg(PersistenceDescriptorVectorizer;
    name="TDAPersistenceDiagrams",uuid="bbe17acd-c4ea-4f01-a2b8-42969f873a90",
    url="https://github.com/JuliaTDA/PersistenceDiagrams.jl",license="MIT",julia=true,is_wrapper=false)
