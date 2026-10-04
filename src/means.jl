"""
    FrechetMeanResult

Diagnostics from [`frechet_mean`](@ref): `diagram`, weighted squared
Wasserstein `objective`, `converged`, `iterations`, monotone `history`, and
optimal `matchings` from the returned diagram to each positive-weight input.
`converged` certifies stationarity of the selected assignment/update run, not
global optimality or uniqueness.
"""
struct FrechetMeanResult
    diagram::PersistenceDiagram
    objective::Float64
    converged::Bool
    iterations::Int
    history::Vector{Float64}
    matchings::Vector{Matching}
end

function _mean_inputs(diagrams, weights)
    diagrams = collect(diagrams)
    isempty(diagrams) && throw(ArgumentError("at least one diagram is required"))
    ws = isnothing(weights) ? ones(length(diagrams)) : Float64.(collect(weights))
    length(ws) == length(diagrams) || throw(DimensionMismatch("one weight per diagram is required"))
    all(w -> isfinite(w) && w >= 0, ws) && any(>(0), ws) ||
        throw(ArgumentError("weights must be finite, nonnegative, and have positive sum"))
    keep = findall(>(0), ws)
    diagrams, ws = diagrams[keep], ws[keep]
    # Normalize without overflow when all weights are large.
    ws ./= maximum(ws)
    ws ./= sum(ws)
    dims = [hasproperty(d, :dim) ? dim(d) : missing for d in diagrams]
    known_dims = filter(d -> !ismissing(d), dims)
    all(d -> isequal(d, first(known_dims)), known_dims) ||
        throw(ArgumentError("diagrams must have the same homology dimension metadata"))
    meta = isempty(known_dims) ? NamedTuple() : (; dim=first(known_dims))
    finite = PersistenceDiagram[]
    essential = Vector{Float64}[]
    for d in diagrams
        for x in d
            isfinite(birth(x)) && !isnan(death(x)) && death(x) >= birth(x) ||
                throw(ArgumentError("intervals need finite births and deaths >= births"))
        end
        push!(finite, PersistenceDiagram(sort([x for x in d if isfinite(x) && persistence(x) > 0])))
        push!(essential, sort([birth(x) for x in d if !isfinite(x)]))
    end
    all(e -> length(e) == length(first(essential)), essential) ||
        throw(ArgumentError("essential intervals must have equal cardinality in every positive-weight diagram"))
    return finite, essential, ws, meta
end

_mean_cost(y, ds, ws) = sum(ws[i] * Wasserstein(2, 2)(y, ds[i])^2 for i in eachindex(ds))

function _mean_update(y, ds, ws)
    n = length(y)
    mids, heights, mass = zeros(n), zeros(n), zeros(n)
    additional = PersistenceInterval[]
    for (d, w) in zip(ds, ws)
        match = Wasserstein(2, 2)(y, d; matching=true)
        for (i, j) in match.matching
            if i <= n && j <= length(d)
                mids[i] += w * midlife(d[j])
                heights[i] += w * persistence(d[j]) / 2
                mass[i] += w
            elseif i > n && j <= length(d)
                # A source point matched to the diagonal can improve the
                # objective by creating a new mean point. Without this step,
                # an empty initialization would incorrectly appear stationary.
                middle = midlife(d[j])
                height = w * persistence(d[j]) / 2
                push!(additional, PersistenceInterval(middle - height, middle + height))
            end
        end
    end
    # Diagonal matches move along with the new center. Their normal component
    # is zero; their tangent component cancels from the normal equations.
    intervals = PersistenceInterval[]
    for i in 1:n
        mass[i] > 0 && heights[i] > 0 || continue
        middle = mids[i] / mass[i]
        push!(intervals, PersistenceInterval(middle - heights[i], middle + heights[i]))
    end
    append!(intervals, additional)
    return PersistenceDiagram(sort(intervals))
end

function _mean_run(seed, ds, ws, max_iterations, atol, rtol)
    y = seed
    history = [_mean_cost(y, ds, ws)]
    isfinite(first(history)) || throw(ArgumentError("endpoint magnitudes cause a nonfinite squared Wasserstein objective"))
    converged = false
    iterations = 0
    for iteration in 1:max_iterations
        candidate = _mean_update(y, ds, ws)
        objective = _mean_cost(candidate, ds, ws)
        isfinite(objective) || throw(ArgumentError("nonfinite squared Wasserstein objective"))
        slack = atol + rtol * max(1.0, last(history))
        objective <= last(history) + slack || error("Fréchet objective increased during an exact assignment/update step")
        movement = Wasserstein(2, 2)(y, candidate)
        y = candidate
        push!(history, objective)
        iterations = iteration
        if movement <= atol + rtol * max(1.0, sqrt(objective))
            converged = true
            break
        end
    end
    return y, history, converged, iterations
end

# A weighted point on a two-diagram Wasserstein geodesic is a global mean of
# those two inputs. Include it among deterministic starts for larger samples.
function _two_diagram_seed(left, right, w)
    intervals = PersistenceInterval[]
    for (x, y) in matching(Wasserstein(2, 2)(left, right; matching=true); bottleneck=false)
        # Diagonal endpoints are projected from the matched off-diagonal point.
        b, d = w * birth(x) + (1 - w) * birth(y), w * death(x) + (1 - w) * death(y)
        b < d && push!(intervals, PersistenceInterval(b, d))
    end
    return PersistenceDiagram(sort(intervals))
end

"""
    frechet_mean(diagrams; weights=nothing, init=nothing, max_iterations=200,
                 atol=1e-10, rtol=1e-8) -> FrechetMeanResult

Minimize `sum(weights[i] * Wasserstein(2, 2)(mean, diagrams[i])^2)` using exact
Hungarian assignments and closed-form, diagonal-aware Euclidean updates
(Turner et al., 2014). Weights are normalized to sum to one. This uses the
Euclidean ground norm, whereas `Wasserstein(2)` defaults to the infinity norm.
Unmatched source points spawn new centers from the diagonal, so an undersized
or empty initialization can increase its cardinality.

By default, run from each positive-weight input and their concatenated finite
support, selecting the smallest objective. Two inputs additionally use their
optimal Wasserstein geodesic midpoint (or weighted interpolation). `init`
selects a single user-provided starting diagram. Assignment descent converges
to a local minimum; multiple starts do not guarantee a global optimum for
three or more inputs. Ties are resolved deterministically by input order.

Zero-weight inputs are excluded. Diagonal points have no effect. Essential
intervals are supported when all positive-weight inputs have the same number:
their sorted birth times are averaged separately. Mixed homology dimensions,
invalid endpoints, and unequal essential cardinalities are rejected. Empty
diagrams are supported. No input is mutated.

```julia
ds = [PersistenceDiagram([(0.0, 2.0)]), PersistenceDiagram([(0.0, 4.0)])]
result = frechet_mean(ds)
result.diagram                    # [(0, 3)]
result.objective, result.converged # 1.0, true
```

Reference: https://arxiv.org/abs/1206.2790.
"""
function frechet_mean(diagrams; weights=nothing, init=nothing, max_iterations=200,
                       atol=1e-10, rtol=1e-8)
    max_iterations isa Integer && max_iterations > 0 ||
        throw(ArgumentError("max_iterations must be a positive integer"))
    all(t -> isfinite(t) && t >= 0, (atol, rtol)) ||
        throw(ArgumentError("tolerances must be finite and nonnegative"))
    ds, essential, ws, meta = _mean_inputs(diagrams, weights)
    seeds = if isnothing(init)
        vcat(ds, [PersistenceDiagram(reduce(vcat, [d.intervals for d in ds]))])
    else
        validated, _, _, _ = _mean_inputs([init], nothing)
        validated
    end
    if isnothing(init) && length(ds) == 2
        push!(seeds, _two_diagram_seed(ds[1], ds[2], ws[1]))
    end
    best = nothing
    for seed in seeds
        candidate = _mean_run(seed, ds, ws, max_iterations, atol, rtol)
        if isnothing(best) || last(candidate[2]) < last(best[2])
            best = candidate
        end
    end
    y, history, converged, iterations = best
    e_births = [sum(ws[i] * essential[i][j] for i in eachindex(ds))
                for j in eachindex(first(essential))]
    e_cost = sum(ws[i] * sum((essential[i] .- e_births).^2) for i in eachindex(ds))
    isfinite(e_cost) || throw(ArgumentError("essential birth magnitudes cause a nonfinite objective"))
    diagram = PersistenceDiagram(vcat(y.intervals,
        [PersistenceInterval(b, Inf) for b in e_births]); meta...)
    original = [PersistenceDiagram(vcat(ds[i].intervals,
        [PersistenceInterval(b, Inf) for b in essential[i]]); meta...) for i in eachindex(ds)]
    matches = [Wasserstein(2, 2)(diagram, d; matching=true) for d in original]
    return FrechetMeanResult(diagram, last(history) + e_cost, converged, iterations,
        history .+ e_cost, matches)
end

"""
    diagram_mean(diagrams; kwargs...) -> PersistenceDiagram

Return the diagram from [`frechet_mean`](@ref). Use `frechet_mean` when
convergence diagnostics or the attained objective are needed.
"""
diagram_mean(diagrams; kwargs...) = frechet_mean(diagrams; kwargs...).diagram

"""
    frechet_variance(diagrams; center=nothing, weights=nothing, kwargs...)

Weighted mean squared Euclidean Wasserstein distance to `center`. When no
center is supplied, compute a [`frechet_mean`](@ref) using `kwargs`.
"""
function frechet_variance(diagrams; center=nothing, weights=nothing, kwargs...)
    diagrams = collect(diagrams)
    isnothing(center) && return frechet_mean(diagrams; weights=weights, kwargs...).objective
    ds, essential, ws, meta = _mean_inputs(diagrams, weights)
    ys, ey, _, center_meta = _mean_inputs([center], nothing)
    (!haskey(meta, :dim) || !haskey(center_meta, :dim) || isequal(meta.dim, center_meta.dim)) ||
        throw(ArgumentError("center must have the same homology dimension metadata"))
    length(first(ey)) == length(first(essential)) || return Inf
    return _mean_cost(first(ys), ds, ws) +
        sum(ws[i] * sum((essential[i] .- first(ey)).^2) for i in eachindex(ds))
end
