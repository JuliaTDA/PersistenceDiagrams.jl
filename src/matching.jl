# Tried to approximate the approach from
# https://mrzv.org/publications/geometry-helps-distances-persistence-diagrams/alenex/ and
# https://www2.cs.arizona.edu/~alon/papers/match.pdf with NearestNeighbors.jl but the
# allocations were so high, it was slower for moderately sized inputs. This approach is also
# much much simpler.
# TODO: try it again some time.

abstract type MatchingDistance end

"""
    weight(::MatchingDistance, left, right)
    weight(::Matching)

Get the weight of the matching between persistence diagrams `left` and `right`.

# See also

* [`matching`](@ref)
* [`Bottleneck`](@ref)
* [`Wasserstein`](@ref)
"""
weight(dist::MatchingDistance, left, right) = dist(left, right; matching=false)
"""
    matching(::MatchingDistance, left, right)
    matching(::Matching)

Get the matching between persistence diagrams `left` and `right`.

# See also

* [`weight`](@ref)
* [`Bottleneck`](@ref)
* [`Wasserstein`](@ref)
"""
matching(dist::MatchingDistance, left, right) = dist(left, right; matching=true)

"""
    Matching

A matching between two persistence diagrams.

# Methods

* [`weight(::Matching)`](@ref)
* [`matching(::Matching)`](@ref)
"""
struct Matching
    left::PersistenceDiagram
    right::PersistenceDiagram
    weight::Float64
    matching::Vector{Pair{Int,Int}}
    bottleneck::Bool
end

weight(match::Matching) = match.weight

Base.length(match::Matching) = length(match.matching)
Base.isempty(match::Matching) = isempty(match.matching)

function _distance(int1, int2, q=Inf)
    diff_birth = abs(birth(int1) - birth(int2))
    if isfinite(int1) && isfinite(int2)
        diff_death = abs(death(int1) - death(int2))
    elseif !isfinite(int1) && !isfinite(int2)
        diff_death = 0.0
    else
        diff_death = Inf
    end
    if q === Inf
        return max(diff_birth, diff_death)
    else
        return (diff_birth^q + diff_death^q)^(1 / q)
    end
end

function _distances(left, right, q=Inf)
    dists = zeros(length(right), length(left))
    for j in eachindex(left), i in eachindex(right)
        dists[i, j] = _distance(left[j], right[i], q)
    end
    return dists
end

function matching(match::Matching; bottleneck=match.bottleneck)
    result = Pair{PersistenceInterval,PersistenceInterval}[]
    n = length(match.left)
    m = length(match.right)
    for (i, j) in match.matching
        if i ≤ n && j ≤ m
            push!(result, match.left[i] => match.right[j])
        elseif i ≤ n
            # left is matched to diagonal
            l = match.left[i]
            push!(result, l => _diagonal_interval(l))
        elseif j ≤ m
            # right is matched to diagonal
            r = match.right[j]
            push!(result, _diagonal_interval(r) => r)
        end
    end
    sort!(result)

    if !bottleneck
        return result
    else
        return filter!(m -> _distance(m...) == match.weight, result)
    end
end

function Base.summary(io::IO, match::Matching)
    b = match.bottleneck ? "Bottleneck " : ""
    return print(io, "$(b)Matching with weight $(match.weight)")
end
function Base.show(io::IO, match::Matching)
    return Base.summary(io, match)
end
function Base.show(io::IO, ::MIME"text/plain", match::Matching)
    print(io, match)
    if length(match) > 0
        print(io, ":")
        pairs = matching(match)
        for p in pairs
            print(io, "\n ", p)
        end
    end
end

_diagonal_interval((b, d)) = PersistenceInterval((b + d) / 2, (b + d) / 2)

"""
    _adjacency_matrix(left::PersistenceDiagram, right::PersistenceDiagram, power)

Get the adjacency matrix of the matching between `left` and `right`. Edge weights are equal
to distances between intervals raised to the power of `power`. Distances between diagonal
points and values that should not be matched with them are set to `Inf`. The same
holds for distances between finite and infinite intervals.

For `length(left) == n` and `length(right) == m`, it returns a ``(n m) × (m n)`` matrix.

# Example

```jldoctest
julia> left = PersistenceDiagram([(0.0, 1.0), (3.0, 4.5)]);

julia> right = PersistenceDiagram([(0.0, 1.0), (4.0, 5.0), (4.0, 7.0)]);

julia> PersistenceDiagrams._adjacency_matrix(left, right)
5×5 Matrix{Float64}:
  0.0   3.5    0.5  Inf   Inf
  4.0   1.0   Inf    0.5  Inf
  6.0   2.5   Inf   Inf    1.5
  0.5  Inf     0.0   0.0   0.0
 Inf    0.75   0.0   0.0   0.0
```
"""
function _adjacency_matrix(left, right, power=1, q=Inf)
    n = length(left)
    m = length(right)
    adj = fill(Inf, n + m, m + n)

    dists = _distances(left, right, q)
    adj[axes(dists)...] .= dists

    for i in 1:n
        adj[i + m, i] = _distance(left[i], _diagonal_interval(left[i]), q)
    end
    for j in 1:m
        adj[j, j + n] = _distance(right[j], _diagonal_interval(right[j]), q)
    end
    adj[(m + 1):(m + n), (n + 1):(n + m)] .= 0.0

    if power ≠ 1
        adj .^= power
    end
    return adj
end

"""
    BottleneckGraph

Representation of the bipartite graph used for computing bottleneck distance via the
Hopcroft-Karp algorithm. In all the following functions, `left` and `right` refer to the
vertex sets of the graph. The graph has `n + m` vertices in each set corresponding to the
numbers of points in the diagrams plus the diagonals.

# Fields

* `adj::Matrix{Float64}`: the adjacency matrix.
* `match_left::Vector{Int}`: matches of left vertices.
* `match_right::Vector{Int}`: matches of right vertices.
* `edges::Vector{Float64}`: edge lengths, unique and sorted.
* `n::Int`: number of intervals in left diagram.
* `m::Int`: number of intervals in right diagram.
"""
struct BottleneckGraph
    adj::Matrix{Float64}

    match_left::Vector{Int}
    match_right::Vector{Int}

    edges::Vector{Float64}

    n_vertices::Int
end

function BottleneckGraph(left::PersistenceDiagram, right::PersistenceDiagram)
    n = length(left)
    m = length(right)
    adj = _adjacency_matrix(left, right)

    edges = filter!(isfinite, sort!(unique!(copy(vec(adj)))))

    return BottleneckGraph(adj, fill(0, n + m), fill(0, m + n), edges, n + m)
end

function _left_neighbors!(buff, graph::BottleneckGraph, vertices, ε, pred)
    empty!(buff)
    for l in vertices
        for r in axes(graph.adj, 1)
            graph.adj[r, l] ≤ ε && pred(r) && push!(buff, r)
        end
    end
    return unique!(buff)
end

function _right_neighbors!(buff, graph::BottleneckGraph, vertices)
    empty!(buff)
    for r in vertices
        push!(buff, graph.match_right[r])
    end
    return buff
end

_is_exposed_right(graph::BottleneckGraph, r) = graph.match_right[r] == 0
_exposed_left(graph::BottleneckGraph) = findall(iszero, graph.match_left)

"""
    _depth_layers(graph::BottleneckGraph, ε)

Split `graph` into layers by how deep they are from a bfs starting at exposed left
vertices in `graph` only taking into account edges of length smaller than or equal to `ε`.
Return depts of right vertices and maximum depth reached.
"""
function _depth_layers(graph::BottleneckGraph, ε)
    depths = fill(0, graph.n_vertices)
    visited = fill(false, graph.n_vertices)
    lefts = _exposed_left(graph)
    rights = Int[]
    i = 1
    while true
        _left_neighbors!(rights, graph, lefts, ε, r -> !visited[r])
        visited[rights] .= true
        depths[rights] .= i
        if isempty(rights)
            # no augmenting path exists
            return nothing, nothing
        elseif any(r -> _is_exposed_right(graph, r), rights)
            return depths, i
        else
            _right_neighbors!(lefts, graph, rights)
        end
        i += 1
    end
end

"""
    _augmenting_paths(graph::BottleneckGraph, ε)

find a maximal set of augmenting paths in graph, taking only edges with weight less than or
equal to `ε` into account.
"""
function _augmenting_paths(graph::BottleneckGraph, ε)
    depths, max_depth = _depth_layers(graph, ε)
    paths = Vector{Int}[]
    isnothing(depths) && return paths

    prev = fill(0, graph.n_vertices)
    rights = Int[]
    lefts = Int[]
    stack = Tuple{Int,Int}[]

    for l_start in _exposed_left(graph)
        empty!(stack)
        push!(stack, (l_start, 1))
        prev .= 0

        while !isempty(stack)
            l, i = pop!(stack)
            parent = graph.match_left[l]
            _left_neighbors!(rights, graph, l, ε, r -> depths[r] == i)
            if i < max_depth
                prev[rights] .= l
                _right_neighbors!(lefts, graph, rights)
                append!(stack, (l, i + 1) for l in lefts)
            else
                found_path = false
                for r in rights
                    if _is_exposed_right(graph, r)
                        prev[r] = l
                        path = Int[r]
                        depths[r] = 0

                        while (l = prev[r]) ≠ l_start
                            @assert prev[r] ≠ 0
                            r = graph.match_left[l]
                            depths[r] = 0
                            append!(path, (l, r))
                        end
                        push!(path, l_start)
                        reverse!(path)
                        push!(paths, path)
                        found_path = true
                        break
                    end
                end
                found_path && break
            end
        end
    end

    return paths
end

function _unmatch_all!(graph::BottleneckGraph)
    graph.match_left .= 0
    graph.match_right .= 0
    return (0, 0)
end

function _augment!(graph, p)
    for i in 1:2:(length(p) - 1)
        l, r = p[i], p[i + 1]
        graph.match_left[l] = r
        graph.match_right[r] = l
    end
end

function _hopcroft_karp!(graph, ε)
    _unmatch_all!(graph)
    paths = _augmenting_paths(graph, ε)
    while !isempty(paths)
        for p in paths
            _augment!(graph, p)
        end
        paths = _augmenting_paths(graph, ε)
    end
    matching = [
        i => graph.match_left[i] for i in 1:(graph.n_vertices) if graph.match_left[i] ≠ 0
    ]
    is_perfect = length(matching) == graph.n_vertices

    return matching, is_perfect
end

"""
    Bottleneck

Use this object to find the bottleneck distance or matching between persistence diagrams.
The distance value is equal to

```math
W_\\infty(X, Y) = \\inf_{\\eta:X\\rightarrow Y} \\sup_{x\\in X} ||x-\\eta(x)||_\\infty,
```

where ``X`` and ``Y`` are the persistence diagrams and ``\\eta`` is a perfect matching
between the intervals. Note the ``X`` and ``Y`` don't need to have the same number of
points, as the diagonal points are considered in the matching as well.

!!! warning
Computing the bottleneck distance requires ``\\mathcal{O}(n^2)`` space and
``\\mathcal{O}(n^3)time``. Be careful when computing distances between very large diagrams!

# Usage

* `Bottleneck()(left, right[; matching=false])`: find the bottleneck matching (if
  `matching=true`) or distance (if `matching=false`) between persistence diagrams `left` and
  `right`

# Example

```jldoctest
julia> left = PersistenceDiagram([(1.0, 2.0), (5.0, 8.0)]);

julia> right = PersistenceDiagram([(1.0, 2.0), (3.0, 4.0), (5.0, 10.0)]);

julia> Bottleneck()(left, right)
2.0

julia> Bottleneck()(left, right; matching=true)
Bottleneck Matching with weight 2.0:
 [5.0, 8.0) => [5.0, 10.0)

```
"""
struct Bottleneck <: MatchingDistance end

function (::Bottleneck)(left::PersistenceDiagram, right::PersistenceDiagram; matching=false)
    if count(!isfinite, left) ≠ count(!isfinite, right)
        if matching
            return Matching(left, right, Inf, Pair{Int,Int}[], true)
        else
            return Inf
        end
    end

    if length(left) == 0 & length(right) == 0
        if matching
            return Matching(left, right, 0, Pair{Int,Int}[], true)
        else
            return 0.0
        end
    end

    graph = BottleneckGraph(left, right)
    edges = graph.edges

    lo = 1
    hi = length(edges)
    while lo < hi - 1
        m = lo + ((hi - lo) >>> 0x01)
        _, succ = _hopcroft_karp!(graph, edges[m])
        if succ
            hi = m
        else
            lo = m
        end
    end
    match, succ = _hopcroft_karp!(graph, edges[lo])
    distance = edges[lo]
    if !succ
        distance = edges[hi]
        match, _ = _hopcroft_karp!(graph, edges[hi])
    end
    @assert length(match) == length(left) + length(right)
    if matching
        return Matching(left, right, distance, match, true)
    else
        return distance
    end
end

function (b::Bottleneck)(left, right; matching=false)
    if length(left) ≠ length(right)
        throw(ArgumentError("`left` and `right` must have the same length"))
    end
    results = (b(l, r; matching=matching) for (l, r) in zip(left, right))
    if matching
        return collect(results)
    else
        return maximum(results)
    end
end

"""
    Wasserstein(p=1, q=Inf)

Use this object to find the Wasserstein distance or matching between persistence diagrams.
The distance value is equal to

```math
W_{p,q}(X,Y)=\\left[\\inf_{\\eta:X\\rightarrow Y}\\sum_{x\\in X}||x-\\eta(x)||_\\q^p\\right]^{1/p},
```

where ``X`` and ``Y`` are the persistence diagrams and ``\\eta`` is a perfect matching
between the intervals. Note the ``X`` and ``Y`` don't need to have the same number of
points, as the diagonal points are considered in the matching as well.

!!! warning
Computing the Wasserstein distance requires ``\\mathcal{O}(n^2)`` space and
``\\mathcal{O}(n^3)`` time. Be careful when computing distances between very large diagrams!

# Usage

* `Wasserstein(p=1, q=Inf)(left, right[; matching=false])`: find the Wasserstein matching
  (if `matching=true`) or distance (if `matching=false`) between persistence diagrams `left`
  and `right`.

# Example

```jldoctest
julia> left = PersistenceDiagram([(1.0, 2.0), (5.0, 8.0)]);

julia> right = PersistenceDiagram([(1.0, 2.0), (3.0, 4.0), (5.0, 10.0)]);

julia> Wasserstein()(left, right)
2.5

julia> Wasserstein()(left, right; matching=true)
Matching with weight 2.5:
 [1.0, 2.0) => [1.0, 2.0)
 [3.5, 3.5) => [3.0, 4.0)
 [5.0, 8.0) => [5.0, 10.0)

```
"""
struct Wasserstein <: MatchingDistance
    p::Float64
    q::Float64

    Wasserstein(p=1, q=Inf) = new(Float64(p), Float64(q))
end

function (w::Wasserstein)(
    left::PersistenceDiagram, right::PersistenceDiagram; matching=false
)
    if length(left) == 0 & length(right) == 0
        if matching
            return Matching(left, right, 0, Pair{Int,Int}[], false)
        else
            return 0.0
        end
    end

    if count(!isfinite, left) == count(!isfinite, right)
        adj = _adjacency_matrix(right, left, w.p, w.q)
        match = collect(i => j for (i, j) in enumerate(hungarian(adj)[1]))

        distance = sum(adj[i, j] for (i, j) in match)^(1 / w.p)

        if matching
            return Matching(left, right, distance, match, false)
        else
            return distance
        end
    else
        if matching
            return Matching(left, right, Inf, Pair{Int,Int}[], false)
        else
            return Inf
        end
    end
end

function (w::Wasserstein)(left, right; matching=false)
    if length(left) ≠ length(right)
        throw(ArgumentError("`left` and `right` must have the same length"))
    end
    results = (w(l, r; matching=matching) for (l, r) in zip(left, right))
    if matching
        return collect(results)
    else
        return sum(results)
    end
end

"""
    SlicedWasserstein(; slices=50)

Use this object to compute the approximate Sliced Wasserstein distance between persistence
diagrams. Unlike [`Bottleneck`](@ref) and [`Wasserstein`](@ref), the sliced Wasserstein
distance is not based on a matching, but on averaging one-dimensional Wasserstein-1 distances
between projections of the diagrams onto a family of lines.

The (true) sliced Wasserstein distance is defined as

```math
SW(X, Y) = \\frac{1}{\\pi} \\int_{-\\pi/2}^{\\pi/2}
    W_1\\bigl(\\mu_X^\\theta + \\Pi_\\Delta(\\mu_Y^\\theta),\\;
             \\mu_Y^\\theta + \\Pi_\\Delta(\\mu_X^\\theta)\\bigr)\\, \\mathrm{d}\\theta,
```

where ``\\mu_X^\\theta`` is the projection of the points of diagram ``X`` onto the line of
angle ``\\theta``, ``\\Pi_\\Delta`` is the orthogonal projection onto the diagonal, and
``W_1`` is the one-dimensional Wasserstein-1 distance. Projecting each diagram's diagonal
projections of the *other* diagram's points augments both diagrams to equal cardinality, so
the one-dimensional optimal transport reduces to sorting.

This implementation approximates the integral with a deterministic Riemann-style average over
`slices` directions ``\\theta_i`` evenly spaced in ``[-\\pi/2, \\pi/2)``. The result is
therefore reproducible: two calls with the same `slices` always return the same value.

!!! note
    The sliced Wasserstein distance is a distance only; it has no associated matching.
    Calling [`matching`](@ref) on it throws an error.

!!! note
    Infinite intervals are not supported and are ignored, consistent with the vectorization
    methods (see [`PersistenceCurve`](@ref)). This differs from [`Bottleneck`](@ref) and
    [`Wasserstein`](@ref), which return `Inf` when the diagrams have a different number of
    infinite intervals.

# Usage

* `SlicedWasserstein(; slices=50)(left, right)`: compute the sliced Wasserstein distance
  between persistence diagrams `left` and `right`.

# Example

```jldoctest
julia> left = PersistenceDiagram([(1.0, 2.0), (5.0, 8.0)]);

julia> right = PersistenceDiagram([(1.0, 2.0), (3.0, 4.0), (5.0, 10.0)]);

julia> round(SlicedWasserstein()(left, right); digits=4)
0.6905

```

# Reference

Carrière, M., Cuturi, M., & Oudot, S. (2017). Sliced Wasserstein kernel for persistence
diagrams. In *Proceedings of the 34th International Conference on Machine Learning (ICML)*,
PMLR 70:664-673. [arXiv:1706.03358](https://arxiv.org/abs/1706.03358).
"""
struct SlicedWasserstein <: MatchingDistance
    slices::Int

    function SlicedWasserstein(; slices=50)
        if slices < 1
            throw(ArgumentError("`slices` must be positive"))
        end
        return new(Int(slices))
    end
end

# Project the (birth, death) point of `int` and the projection of `int` onto the diagonal
# onto the line through the origin with angle θ. `(cosθ, sinθ)` is the unit direction.
_project_point(int, cosθ, sinθ) = birth(int) * cosθ + death(int) * sinθ
function _project_diagonal(int, cosθ, sinθ)
    m = midlife(int)
    return m * cosθ + m * sinθ
end

function (sw::SlicedWasserstein)(
    left::PersistenceDiagram, right::PersistenceDiagram; matching=false
)
    if matching
        throw(ArgumentError("the sliced Wasserstein distance has no matching"))
    end

    # Infinite intervals are unsupported; ignore them (see docstring).
    l = filter(isfinite, left)
    r = filter(isfinite, right)

    n = length(l)
    m = length(r)
    if n == 0 && m == 0
        return 0.0
    end

    # For each direction, both diagrams are augmented with the diagonal projections of the
    # other diagram's points, so both sorted sequences have length `n + m`.
    proj1 = Vector{Float64}(undef, n + m)
    proj2 = Vector{Float64}(undef, n + m)

    total = 0.0
    for k in 0:(sw.slices - 1)
        θ = -π / 2 + k * (π / sw.slices)
        cosθ = cos(θ)
        sinθ = sin(θ)

        # `proj1` holds left's projected points (1:n) and right's diagonal projections
        # (n+1:n+m); `proj2` holds right's projected points (1:m) and left's diagonal
        # projections (m+1:m+n). This is the standard augmentation that makes both
        # sequences have length n + m.
        for i in 1:n
            proj1[i] = _project_point(l[i], cosθ, sinθ)
            proj2[m + i] = _project_diagonal(l[i], cosθ, sinθ)
        end
        for j in 1:m
            proj2[j] = _project_point(r[j], cosθ, sinθ)
            proj1[n + j] = _project_diagonal(r[j], cosθ, sinθ)
        end

        sort!(proj1)
        sort!(proj2)

        s = 0.0
        for idx in eachindex(proj1)
            s += abs(proj1[idx] - proj2[idx])
        end
        total += s
    end

    # Average over directions, with the 1/π normalisation of the continuous definition.
    return total / sw.slices / π
end

function (sw::SlicedWasserstein)(left, right; matching=false)
    if matching
        throw(ArgumentError("the sliced Wasserstein distance has no matching"))
    end
    if length(left) ≠ length(right)
        throw(ArgumentError("`left` and `right` must have the same length"))
    end
    return sum(sw(l, r) for (l, r) in zip(left, right))
end
