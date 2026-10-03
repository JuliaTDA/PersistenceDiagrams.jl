"""
# TDAPersistenceDiagrams.jl

Types and functions for working with persistence diagrams.

Experimental JuliaTDA fork of PersistenceDiagrams.jl. See the documentation in `docs/src`.
"""
module TDAPersistenceDiagrams

export PersistenceDiagram,
    PersistenceInterval,
    birth,
    death,
    persistence,
    midlife,
    representative,
    birth_simplex,
    death_simplex,
    dim,
    threshold,
    Bottleneck,
    Wasserstein,
    SlicedWasserstein,
    weight,
    matching,
    SlicedWassersteinKernel,
    PersistenceFisherKernel,
    kernel_matrix,
    PersistenceImage,
    PersistenceCurve,
    BettiCurve,
    Life,
    Midlife,
    LifeEntropy,
    MidlifeEntropy,
    PDThresholding,
    Landscape,
    Landscapes,
    Silhuette,
    persistence_entropy,
    barcode,
    PersistenceImageVectorizer,
    PersistenceCurveVectorizer,
    PersistenceLandscapeVectorizer

using Compat
using Hungarian
using LinearAlgebra
using RecipesBase
using ScientificTypes
using Statistics
using Tables

include("intervals.jl")
include("diagrams.jl")
include("tables.jl")
include("matching.jl")
include("kernels.jl")

include("persistencecurves.jl")
include("persistenceimages.jl")

include("plotsrecipes.jl")

include("scitypes.jl")
include("mlj.jl")

end
