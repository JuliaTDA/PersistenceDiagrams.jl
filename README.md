# TDAPersistenceDiagrams.jl

[![Build Status](https://github.com/JuliaTDA/PersistenceDiagrams.jl/actions/workflows/Test.yml/badge.svg?branch=master)](https://github.com/JuliaTDA/PersistenceDiagrams.jl/actions/workflows/Test.yml)

Experimental persistence-diagram tools for the [JuliaTDA](https://github.com/JuliaTDA)
ecosystem, derived from [PersistenceDiagrams.jl](https://github.com/mtsch/PersistenceDiagrams.jl)
by Matija Čufar and contributors. The fork is maintained by G. Vituri and
JuliaTDA contributors and starts its own version series at `0.1.0`.

Package UUID: `bbe17acd-c4ea-4f01-a2b8-42969f873a90`.
The GitHub repository currently retains its original
`JuliaTDA/PersistenceDiagrams.jl` slug; the Julia package and local checkout
are named `TDAPersistenceDiagrams` and `TDAPersistenceDiagrams.jl`.
This fork is not registered.

This package provides the `PersistenceInterval` and `PersistenceDiagram` types as well as
some functions for working with them. See the
[documentation sources](docs/src/index.md) for more info.

To compute persistent homology in this ecosystem, use
[TDARipserer.jl](https://github.com/JuliaTDA/Ripserer.jl).

## Fork scope and compatibility

The starting point is upstream PersistenceDiagrams `0.9.10`
(commit `86eed79`). JuliaTDA additions already include `SlicedWasserstein`,
`persistence_entropy`, `SlicedWassersteinKernel`, `PersistenceFisherKernel`
and `kernel_matrix`. Further experimental distances, kernels and
vectorizations can evolve independently of upstream.

Public names such as `PersistenceDiagram` and `Bottleneck` are retained.
Types from this fork are distinct from the original package's types.
TDARipserer, TDAplots, PersistenceInference and JuliaTDA use the fork's types
consistently. API compatibility is experimental and breaking changes
increment the minor version before `1.0`. See [CHANGELOG.md](CHANGELOG.md).

## Local development

From a sibling package environment:

```julia
using Pkg
Pkg.develop(path = "../TDAPersistenceDiagrams.jl")
using TDAPersistenceDiagrams
```

After these changes are pushed, the existing repository URL also works:
`Pkg.develop(url = "https://github.com/JuliaTDA/PersistenceDiagrams.jl")`.

## Tracking upstream and attribution

The Git history and upstream attribution are preserved. Inherited Git tags
describe upstream releases, rather than releases of this independent fork. A local `upstream`
remote points to `https://github.com/mtsch/PersistenceDiagrams.jl.git`.
Fresh clones can add it with
`git remote add upstream https://github.com/mtsch/PersistenceDiagrams.jl.git`.
Use `git fetch --no-tags upstream` to inspect changes, then selectively merge
or cherry-pick relevant fixes. Preserve the fork's name, UUID and MLJ
metadata, and port changes to `src/TDAPersistenceDiagrams.jl`. Run the package
tests and downstream integration tests after syncing.

The original MIT copyright and license are retained in [LICENSE](LICENSE),
with an additional copyright notice for JuliaTDA contributions.
