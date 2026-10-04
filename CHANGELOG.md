# TDAPersistenceDiagrams v0.1.0 (unreleased)

* Correct the Sliced Wasserstein quadrature normalization (remove a second 1/π
  factor), changing distance and SW kernel values. Add pinned GUDHI numerical
  fixtures, benchmark scripts, baseline measurements and documented tolerances.
  Multiply old kernel bandwidths by √π to preserve their kernel values.

* Add recommended MLJ hyperparameter metadata and pipeline-aware `tuning_ranges`.
  Handle training folds with no finite bars without consulting held-out data.

* Add persistence scale-space and persistence weighted Gaussian kernels, including optional outer Gaussian embedding kernels.
* Add weighted Euclidean Wasserstein Fréchet means, diagram averaging, variance and convergence diagnostics with essential-interval support.
* Fix asymmetric empty-diagram handling in Bottleneck and Wasserstein (bitwise operator precedence previously returned zero for some nonempty comparisons).

* Add Euler characteristic curves, integrated persistence blocks, tropical
  coordinates, compact tent templates, complex polynomials, topological vectors,
  and seeded ATOL codebooks, with fixed-shape MLJ adapters and reference tests.

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
