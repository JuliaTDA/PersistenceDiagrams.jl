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
    PersistenceScaleSpaceKernel,
    PersistenceWeightedGaussianKernel,
    kernel_matrix,
    FrechetMeanResult,
    frechet_mean,
    diagram_mean,
    frechet_variance,
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

export EulerCharacteristicCurve, PersistenceBlock, TropicalCoordinates, TentTemplate,
    ComplexPolynomial, TopologicalVector, Atol, PersistenceDescriptorVectorizer
export recommended_ranges, tuning_ranges

using Compat
using Hungarian
using LinearAlgebra
using RecipesBase
using Random
using ScientificTypes
using Statistics
using Tables

include("intervals.jl")
include("diagrams.jl")
include("tables.jl")
include("matching.jl")
include("kernels.jl")
include("means.jl")

include("persistencecurves.jl")
include("persistenceimages.jl")
include("descriptors.jl")

include("plotsrecipes.jl")

include("scitypes.jl")
include("mlj.jl")
include("descriptor_mlj.jl")
include("tuning.jl")

end
