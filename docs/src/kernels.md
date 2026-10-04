# Kernels

Positive (semi-)definite kernels turn persistence diagrams into inputs for kernel-based
machine learning, such as support vector machines, kernel PCA, and Gaussian processes. Use
[`kernel_matrix`](@ref) to assemble the Gram matrix over a collection of diagrams.

```@docs
SlicedWassersteinKernel
```

```@docs
PersistenceFisherKernel
```

The scale-space kernel uses a reflected heat kernel to make diagonal points
contribute zero. Its `sigma` is diffusion **time**. The persistence weighted
Gaussian kernel embeds weighted diagram points in a Gaussian RKHS; its `sigma`
is a Gaussian **standard deviation**. Both ignore essential intervals.

```@docs
PersistenceScaleSpaceKernel
PersistenceWeightedGaussianKernel
```

```@example additional_kernels
using TDAPersistenceDiagrams
diagrams = [PersistenceDiagram([(0.0, 2.0)]), PersistenceDiagram([(0.5, 3.0)])]
kernel_matrix(PersistenceScaleSpaceKernel(; sigma=0.5), diagrams)
```

Use `bandwidth=nothing` for the weighted Gaussian RKHS inner product, or a
positive bandwidth for an outer Gaussian of the embedding distance. The linear
forms are unnormalized and have zero similarity with an empty diagram; the
outer Gaussian has a self-similarity of one. A custom weight must vanish on the
diagonal; it does not automatically inherit the stability theorem for the
paper's arctangent persistence weight. See `examples/p3_diagrams.jl`.

```@docs
kernel_matrix
```
