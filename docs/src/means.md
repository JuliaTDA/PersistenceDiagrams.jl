# Diagram averaging

`frechet_mean` computes a weighted Fréchet mean for the **squared Euclidean
Wasserstein distance** `Wasserstein(2, 2)`. This is different from the default
infinity ground norm of `Wasserstein(2)`. The objective is normalized by the
sum of the weights.

Each iteration solves optimal assignments to the current mean, including
matches to the diagonal, then updates the matched groups in closed form.
Diagonal matches reduce a group's persistence without pulling its midpoint
toward an arbitrary fixed diagonal point. The objective decreases along the
assignment/update iterations. The returned history and convergence flag make
termination inspectable.
Source points left unmatched to a mean center create new centers from the
diagonal. This permits cardinality growth from an empty or undersized start;
convergence is not restricted to a preselected number of mean points.

There can be multiple local and global minima. The default performs
deterministic starts from every input and their concatenated support, and
chooses the lowest attained objective. Two inputs also use a point on their
optimal Wasserstein geodesic. For three or more inputs, this is a local
optimization method, not a certificate of global optimality. A user-provided
`init` selects a single start; use it for continuation or comparison studies.

Essential intervals are retained when each input has the same number. Their
sorted birth coordinates are averaged separately; unequal counts are rejected
because the distance would be infinite. Inputs are never mutated. Explicitly
split diagrams of different homology dimensions before averaging.

```@example means
using TDAPersistenceDiagrams
ds = [PersistenceDiagram([(0.0, 2.0)]), PersistenceDiagram([(0.0, 4.0)])]
result = frechet_mean(ds)
[(birth(x), death(x)) for x in result.diagram], result.objective, result.converged
```

The analytic reference corpus in `test/means.jl` includes identical diagrams,
weighted diagonal matches, widely separated points whose mean has increased
cardinality, essential intervals, and a three-diagram closed-form mean. Run
`examples/p3_diagrams.jl` for a larger averaging/kernel workflow.

Reference: Turner, Mileyko, Mukherjee and Harer,
[Fréchet Means for Distributions of Persistence Diagrams](https://arxiv.org/abs/1206.2790).

```@docs
FrechetMeanResult
frechet_mean
diagram_mean
frechet_variance
```
