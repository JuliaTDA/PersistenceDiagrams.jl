# Additional diagram descriptors

The native descriptors are callable objects with fixed output sizes. Fitted grids
and ATOL codebooks must be learned exclusively from training diagrams, then reused
unchanged on held-out data. `PersistenceDescriptorVectorizer` performs this fitting
inside MLJ, including its cross-validation folds.

```@example descriptors
using TDAPersistenceDiagrams
diagrams = [PersistenceDiagram([(0.0,2.0),(0.5,1.5)];dim=1),
            PersistenceDiagram([(0.0,1.5)];dim=1)]
descriptors = (PersistenceBlock(diagrams;size=(3,3)),
               TropicalCoordinates(;order=3),
               TentTemplate(diagrams;size=(3,3)),
               ComplexPolynomial(;length=3),
               TopologicalVector(;length=3),
               Atol(diagrams;n_centers=3))
[v(diagrams[1]) for v in descriptors]
```

| Descriptor | Output | Stability and interpretation |
|---|---|---|
| Euler characteristic curve | One value per time | Alternating sum of supplied Betti numbers; includes essential intervals. Pointwise counts are discontinuous; L1 integration has matched-endpoint control. |
| Persistence block | Exact cell integrals | Squares shrink with persistence. Integration captures short bars and cell-boundary crossings. |
| Tropical coordinates | Two families of `order` coordinates | Symmetric max-plus elementary sums, diagonal invariant, explicit bottleneck Lipschitz bounds. Nonnegative births are required. |
| Tent templates | One sum per template | Fixed compact supports vanish on the diagonal and are Wasserstein-1 stable. |
| Complex polynomial | `2*length` real/imaginary values | S/T root maps vanish on the diagonal; coefficients can be ill-conditioned at high degrees. R does not vanish near the diagonal. No global stability bound is claimed. |
| Topological vector | `length` sorted pair features | Distances capped by both half-persistences; invariant to bar order and diagonal points. |
| ATOL | `n_centers` contrast integrals | Fits a weighted Lloyd codebook, then integrates smooth Gaussian contrasts. Fixed codebook measure stability does not imply bottleneck stability. |

The last three methods were evaluated and included as bounded descriptors.
Polynomial and topological vectors offer inexpensive fixed features; ATOL adapts
to training distributions but adds a codebook-fitting step and requires careful
fold isolation. Empty/small ATOL training dimensions keep the requested output
shape through inactive centers, rather than learning from test data.

ECC sums only the supplied homology dimensions: omitted nonzero homology makes
the result a truncated Euler sum. Native ECC accepts a diagram vector in dimension
order, or uses diagram metadata. Its MLJ adapter requires dimension metadata and
combines all diagram columns in each row into one curve. Other descriptors
concatenate features per diagram column. Infinite bars are excluded except in ECC.

```@docs
EulerCharacteristicCurve
PersistenceBlock
TropicalCoordinates
TentTemplate
ComplexPolynomial
TopologicalVector
Atol
PersistenceDescriptorVectorizer
```

References are linked in the API docstrings. The definitions, analytic examples,
perturbation tests, order invariance, empty cases, output sizes, and MLJ fitting
isolation are covered in `test/descriptors.jl`.
