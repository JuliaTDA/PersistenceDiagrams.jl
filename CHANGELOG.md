# TDAPersistenceDiagrams v0.1.0 (unreleased)

* Establishes an independent experimental JuliaTDA fork of PersistenceDiagrams.jl with its own package name and UUID.
* Preserves the public function/type names; updates imports, MLJ metadata, tests and documentation to the new package identity.
* Preserves the local Sliced Wasserstein, persistence entropy, Sliced Wasserstein kernel and Persistence Fisher kernel additions.

## Inherited upstream release history

# v0.9.10

* Bugfix with Bottleneck and Wasserstein distances. Wasserstein now supports changing the internal norm.

# v0.9.0

* MLJ integration is now exported by default.

# v0.8.1

* Experimental integration with [MLJ.jl](https://github.com/alan-turing-institute/MLJ.jl).
* Distances can now be computed between collections of diagrams.

# v0.8.0

* Split base functionality into
  [PersistenceDiagramsBase.jl](https://github.com/mtsch/PersistenceDiagramsBase.jl).
* `PersistenceDiagram`s and `PersistenceInterval`s are no longer specialized on metadata
  types. This generally makes them easier to work with.
* `PersistenceImage` changes:
  - `slope_end` is now relative to maximum persistence shown in image,
  - default value of `sigma` changed to 2× pixel size (in the larger direction),
  - improved performance.
* Added `Landscapes`.
