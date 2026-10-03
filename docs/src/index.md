# TDAPersistenceDiagrams.jl

An independent experimental JuliaTDA fork of
[PersistenceDiagrams.jl](https://github.com/mtsch/PersistenceDiagrams.jl),
originally authored by Matija Čufar and contributors, and maintained here by
G. Vituri and JuliaTDA contributors. It has its own UUID and version series
and is not registered. See the [fork README](https://github.com/JuliaTDA/PersistenceDiagrams.jl)
for local installation and upstream tracking instructions.

This package provides the `PersistenceInterval` and `PersistenceDiagram` types as well as
some functions for working with them. If you want to compute persistence diagrams, please
see [TDARipserer.jl](https://github.com/JuliaTDA/Ripserer.jl). For examples and tutorials, see
its [documentation sources](https://github.com/JuliaTDA/Ripserer.jl/tree/master/docs/src).

## Overview

This package currently supports the following:

* persistence diagram plotting
* bottleneck, Wasserstein and Sliced Wasserstein distances
* persistence entropy, Sliced Wasserstein and Persistence Fisher kernels, and `kernel_matrix`
* various vectorization methods including persistence images, betti curves, landscapes, and
  more (see [Vectorization](vectorization.md) for full list)
* integration with [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl).
