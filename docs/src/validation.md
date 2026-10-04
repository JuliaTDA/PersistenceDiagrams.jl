# Numerical validation and benchmarks

`validation/gudhi_reference.py` generates fixtures with GUDHI **3.11.0**,
NumPy 2.2.6, SciPy 1.15.3 and scikit-learn 1.7.2, seed **20261003**.
`validation/compare_gudhi.jl` checks 18 cases (empty, unequal-cardinality,
negative-birth and larger diagrams) at 2, 50 and 200 directions. The checked-in
fixtures run in the ordinary test suite without a Python dependency.

The SW comparison uses GUDHI's projection-distance primitive with the same
complete uniform angle grid as Julia. GUDHI 3.11's high-level direction builder
drops the final direction; using identical projections avoids comparing different
quadratures. The SW kernel uses `exp(-SW/(2σ²))`; its parameter corresponds to
GUDHI's exponential bandwidth `2σ²`. Fisher uses Gaussian bandwidth 0.7 and
temperature 1.2, with identical augmented support and discrete normalization.
Empty/empty Fisher similarity is 1 by Julia's documented convention.

Absolute and relative tolerances are **1e-10** for SW/SW kernel and **2e-8**
for Fisher (`acos` amplifies roundoff near 1). The recorded largest errors on
Julia 1.12.5 were 3.55e-15, 2.22e-16 and 2.33e-15 respectively.

This comparison exposed and corrected an extra division by π in the previous
SW implementation. Uniform quadrature of `(1/π) ∫ f(θ)dθ` is `sum(f)/slices`.
SW values are now π times the old values; persisted kernel matrices and tuned
bandwidths should be recomputed. To preserve an old SW kernel value, multiply
its old `bandwidth` by √π. The finite-direction approximation remains dependent
on direction count and orientation; agreement on a quadrature does not prove
an exact continuous-distance error bound.

From the ecosystem workspace, regenerate and compare:

```sh
python3 -m venv /tmp/tda-reference
/tmp/tda-reference/bin/pip install -r TDAPersistenceDiagrams.jl/validation/requirements.txt
/tmp/tda-reference/bin/python TDAPersistenceDiagrams.jl/validation/gudhi_reference.py
julia --project=TDAPersistenceDiagrams.jl/benchmarks -e 'using Pkg; Pkg.develop(PackageSpec(path="TDAPersistenceDiagrams.jl")); Pkg.instantiate()'
julia --project=TDAPersistenceDiagrams.jl/benchmarks TDAPersistenceDiagrams.jl/validation/compare_gudhi.jl
julia --project=TDAPersistenceDiagrams.jl/benchmarks TDAPersistenceDiagrams.jl/benchmarks/suite.jl
```

`benchmarks/suite.jl` warms up every operation, uses `evals=1`, 20 requested
samples, a one-second sampling budget per case, and single-threaded BLAS.
It measures SW, the SW kernel, and SW/Fisher Gram matrices. The checked-in
`baseline.csv` includes median time, allocation bytes/count, actual sample
count, seed, Julia/package versions and thread counts.
The companion `.metadata` file records CPU, operating system, architecture,
Git revision, working-tree state and BenchmarkTools version.
Timings are a baseline for the recorded machine, not a cross-machine performance guarantee. Re-run
under the same settings before comparing changes; compilation is excluded.

Reference implementations: [GUDHI representations](https://gudhi.inria.fr/python/latest/representations.html),
[Carrière et al. (2017)](https://proceedings.mlr.press/v70/carriere17a.html),
[Le & Yamada (2018)](https://arxiv.org/abs/1802.03569).
