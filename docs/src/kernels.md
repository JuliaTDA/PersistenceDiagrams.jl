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

```@docs
kernel_matrix
```
