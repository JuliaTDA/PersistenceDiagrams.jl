# Positive (semi-)definite kernels on persistence diagrams. These turn diagrams into inputs
# for kernel-based machine learning (SVMs, kernel PCA, Gaussian processes, ...), matching the
# functionality offered by GUDHI's `representations` module.

"""
    AbstractPersistenceKernel

Supertype for kernels between persistence diagrams. A kernel `k` is callable as
`k(left, right)` and returns a `Float64` similarity score between the persistence diagrams
`left` and `right`.

# See also

* [`SlicedWassersteinKernel`](@ref)
* [`PersistenceFisherKernel`](@ref)
* [`kernel_matrix`](@ref)
"""
abstract type AbstractPersistenceKernel end

"""
    SlicedWassersteinKernel(; slices=50, bandwidth=1.0)

The sliced Wasserstein kernel between persistence diagrams. It is defined as

```math
k(X, Y) = \\exp\\left(-\\frac{SW(X, Y)}{2\\,\\sigma^2}\\right),
```

where ``SW`` is the (approximate) sliced Wasserstein distance computed with the given number
of `slices` (see [`SlicedWasserstein`](@ref)) and ``\\sigma`` is the `bandwidth`. The
projection logic is shared with [`SlicedWasserstein`](@ref); this kernel only exponentiates
the resulting distance.

!!! note
    Carrière, Cuturi, & Oudot (2017) prove that this kernel is positive semi-definite, so it
    is a valid kernel for support vector machines and other kernel methods.

The kernel value lies in ``(0, 1]``: it equals `1` exactly when the diagrams coincide (the
distance is zero) and decreases towards `0` as the diagrams grow further apart. A smaller
`bandwidth` makes the kernel decay faster, sharpening the distinction between diagrams.

Infinite intervals are ignored, inheriting the behaviour of [`SlicedWasserstein`](@ref).

# Usage

* `SlicedWassersteinKernel(; slices=50, bandwidth=1.0)(left, right)`: compute the sliced
  Wasserstein kernel value between persistence diagrams `left` and `right`.

# Example

```jldoctest
julia> left = PersistenceDiagram([(1.0, 2.0), (5.0, 8.0)]);

julia> right = PersistenceDiagram([(1.0, 2.0), (3.0, 4.0), (5.0, 10.0)]);

julia> SlicedWassersteinKernel()(left, left)
1.0

julia> round(SlicedWassersteinKernel()(left, right); digits=4)
0.708

```

# See also

* [`SlicedWasserstein`](@ref): the underlying distance.
* [`kernel_matrix`](@ref): build the Gram matrix over a collection of diagrams.

# Reference

Carrière, M., Cuturi, M., & Oudot, S. (2017). Sliced Wasserstein kernel for persistence
diagrams. In *Proceedings of the 34th International Conference on Machine Learning (ICML)*,
PMLR 70:664-673. [arXiv:1706.03358](https://arxiv.org/abs/1706.03358).
"""
struct SlicedWassersteinKernel <: AbstractPersistenceKernel
    distance::SlicedWasserstein
    bandwidth::Float64

    function SlicedWassersteinKernel(; slices=50, bandwidth=1.0)
        if bandwidth ≤ 0
            throw(ArgumentError("`bandwidth` must be positive"))
        end
        # `SlicedWasserstein` validates `slices ≥ 1`.
        return new(SlicedWasserstein(; slices=slices), Float64(bandwidth))
    end
end

function (k::SlicedWassersteinKernel)(left, right)
    sw = k.distance(left, right)
    return exp(-sw / (2 * k.bandwidth^2))
end

# Gaussian density (unnormalised, the normalisation cancels because the smoothed measures are
# rescaled to sum to one over the support set) of `x` centred at `μ` with standard deviation
# `σ`, where `x` and `μ` are `(birth, death)` points.
function _gaussian(bx, dx, bμ, dμ, σ)
    return exp(-((bx - bμ)^2 + (dx - dμ)^2) / (2 * σ^2))
end

# Evaluate the normalised, Gaussian-smoothed measure of the diagram `points` (a vector of
# `(birth, death)` tuples) on the support set `support`, returning a probability vector that
# sums to one. An empty diagram has no mass, which the caller handles separately.
function _smoothed_measure(points, support, σ)
    ρ = Vector{Float64}(undef, length(support))
    for (i, (bx, dx)) in enumerate(support)
        s = 0.0
        for (bμ, dμ) in points
            s += _gaussian(bx, dx, bμ, dμ, σ)
        end
        ρ[i] = s
    end
    total = sum(ρ)
    if total > 0
        ρ ./= total
    end
    return ρ
end

# Diagonal projection of `(b, d)`: the closest point on the diagonal, i.e. the midlife point.
_diagonal_projection((b, d)) = ((b + d) / 2, (b + d) / 2)

"""
    PersistenceFisherKernel(; bandwidth=1.0, sigma=1.0)

The persistence Fisher kernel between persistence diagrams. It is defined as

```math
k(X, Y) = \\exp\\left(-\\frac{d_{FIM}(X, Y)}{t}\\right),
```

where ``t`` is the kernel temperature (the `bandwidth` keyword) and ``d_{FIM}`` is the Fisher
information metric distance between the diagrams.

Each diagram is augmented with the diagonal projections of the *other* diagram's points and
represented as a normalised measure obtained by placing an isotropic Gaussian of standard
deviation ``\\sigma`` (the `sigma` keyword) at every point and rescaling to sum to one over
the shared support set. Writing ``\\rho_X`` and ``\\rho_Y`` for these two probability
vectors, the Fisher information metric distance is

```math
d_{FIM}(X, Y) = \\arccos\\left(\\sum_i \\sqrt{\\rho_{X,i}\\,\\rho_{Y,i}}\\right).
```

The inner product is clamped to ``[0, 1]`` before taking ``\\arccos`` to avoid domain errors
caused by floating-point round-off (the value can otherwise slightly exceed `1` for nearly
identical diagrams).

# Parameters

* `sigma`: the standard deviation ``\\sigma`` of the Gaussians used to smooth the diagrams
  into measures. Larger values blur the diagrams more.
* `bandwidth`: the temperature ``t`` in the exponential. Smaller values make the kernel decay
  faster as the Fisher distance grows.

The kernel value lies in ``(0, 1]`` and equals `1` exactly when the two augmented measures
coincide (in particular, `k(X, X) == 1`).

# Edge cases

* Two empty diagrams have no points to augment, so ``d_{FIM} = 0`` and `k = 1`.
* An empty diagram against a non-empty one is handled by the augmentation: the empty diagram
  contributes only the diagonal projections of the other diagram's points, so both measures
  are well defined and non-empty.

# Usage

* `PersistenceFisherKernel(; bandwidth=1.0, sigma=1.0)(left, right)`: compute the persistence
  Fisher kernel value between persistence diagrams `left` and `right`.

# Example

```jldoctest
julia> left = PersistenceDiagram([(0.0, 1.0)]);

julia> PersistenceFisherKernel()(left, left)
1.0

julia> right = PersistenceDiagram([(0.0, 1.0), (2.0, 5.0)]);

julia> 0 < PersistenceFisherKernel()(left, right) < 1
true

```

# See also

* [`SlicedWassersteinKernel`](@ref): another kernel on persistence diagrams.
* [`kernel_matrix`](@ref): build the Gram matrix over a collection of diagrams.

# Reference

Le, T., & Yamada, M. (2018). Persistence Fisher kernel: A Riemannian manifold kernel for
persistence diagrams. In *Advances in Neural Information Processing Systems (NeurIPS)* 31.
[arXiv:1802.03569](https://arxiv.org/abs/1802.03569).
"""
struct PersistenceFisherKernel <: AbstractPersistenceKernel
    bandwidth::Float64
    sigma::Float64

    function PersistenceFisherKernel(; bandwidth=1.0, sigma=1.0)
        if bandwidth ≤ 0
            throw(ArgumentError("`bandwidth` must be positive"))
        end
        if sigma ≤ 0
            throw(ArgumentError("`sigma` must be positive"))
        end
        return new(Float64(bandwidth), Float64(sigma))
    end
end

function (k::PersistenceFisherKernel)(left, right)
    # Infinite intervals are ignored, consistent with the vectorization methods and the
    # sliced Wasserstein distance.
    pts_l = [(birth(int), death(int)) for int in left if isfinite(int)]
    pts_r = [(birth(int), death(int)) for int in right if isfinite(int)]

    if isempty(pts_l) && isempty(pts_r)
        # Both measures are empty (zero mass everywhere); d_FIM = 0 by convention.
        return 1.0
    end

    # Standard augmentation: each diagram also carries the diagonal projections of the other
    # diagram's points, so both measures are supported on a common, non-empty set.
    diag_l = [_diagonal_projection(p) for p in pts_l]
    diag_r = [_diagonal_projection(p) for p in pts_r]

    aug_l = vcat(pts_l, diag_r)
    aug_r = vcat(pts_r, diag_l)

    support = vcat(aug_l, aug_r)

    ρ_l = _smoothed_measure(aug_l, support, k.sigma)
    ρ_r = _smoothed_measure(aug_r, support, k.sigma)

    inner = 0.0
    for i in eachindex(ρ_l)
        inner += sqrt(ρ_l[i] * ρ_r[i])
    end
    inner = clamp(inner, 0.0, 1.0)
    d_fim = acos(inner)

    return exp(-d_fim / k.bandwidth)
end

"""
    kernel_matrix(k, diagrams; symmetric=true)

Compute the Gram (kernel) matrix of the kernel `k` over a collection of persistence
`diagrams`. Entry `(i, j)` is `k(diagrams[i], diagrams[j])`.

When `symmetric=true` (the default), the kernel is assumed symmetric -- as both
[`SlicedWassersteinKernel`](@ref) and [`PersistenceFisherKernel`](@ref) are -- so only the
upper triangle is evaluated (halving the work) and the result is returned as a
`LinearAlgebra.Symmetric` matrix, ready to be passed to kernel methods. With
`symmetric=false`, every entry is computed independently and a plain `Matrix{Float64}` is
returned.

# Example

```jldoctest
julia> diagrams = [
           PersistenceDiagram([(1.0, 2.0)]),
           PersistenceDiagram([(1.0, 2.0), (3.0, 4.0)]),
           PersistenceDiagram([(0.0, 5.0)]),
       ];

julia> G = kernel_matrix(SlicedWassersteinKernel(), diagrams);

julia> size(G)
(3, 3)

julia> all(G[i, i] == 1 for i in 1:3)
true

julia> G == G'
true

```

# See also

* [`SlicedWassersteinKernel`](@ref)
* [`PersistenceFisherKernel`](@ref)
"""
function kernel_matrix(k::AbstractPersistenceKernel, diagrams; symmetric=true)
    n = length(diagrams)
    diagrams = collect(diagrams)
    G = Matrix{Float64}(undef, n, n)
    if symmetric
        for j in 1:n
            for i in 1:j
                v = k(diagrams[i], diagrams[j])
                G[i, j] = v
                G[j, i] = v
            end
        end
        return Symmetric(G)
    else
        for j in 1:n, i in 1:n
            G[i, j] = k(diagrams[i], diagrams[j])
        end
        return G
    end
end
