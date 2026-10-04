# Fixed descriptors and learned codebooks. Every fit uses training diagrams only.

function _finite_points(diagram)
    all(i -> isfinite(birth(i)) && !isnan(death(i)) && death(i) >= birth(i), diagram) ||
        throw(ArgumentError("intervals need finite births and deaths >= births"))
    return [(birth(i), death(i)) for i in diagram if isfinite(i) && persistence(i) > 0]
end

function _descriptor_limits(diagrams)
    points = reduce(vcat, _finite_points.(diagrams); init=Tuple{Float64,Float64}[])
    isempty(points) && return ((0.0,1.0),(0.0,1.0))
    lo, hi = extrema(first.(points))
    padding = 0.05 * max(hi-lo, 1.0)
    return ((lo-padding,hi+padding),(0.0,1.05maximum(d-b for (b,d) in points)))
end

function _descriptor_grid(grid)
    values = Float64.(collect(grid))
    length(values) >= 2 && all(isfinite,values) && all(>(0),diff(values)) ||
        throw(ArgumentError("grid needs at least two finite strictly increasing values"))
    return values
end

"""
    EulerCharacteristicCurve(grid)
    curve(diagrams)

Pointwise Euler characteristic `sum((-1)^q * beta_q(t))` at the fixed `grid`.
Input is a vector of diagrams, one per homology dimension. Dimension metadata
takes precedence; when absent, vector positions correspond to dimensions 0,1,….
Essential intervals are counted, using half-open intervals `birth <= t < death`.
A single diagram contributes its signed Betti curve (dimension defaults to zero).
This is the Euler characteristic of the supplied homology dimensions; include
all dimensions with nonzero homology for a complete ECC. Pointwise counts are
not bottleneck-stable; matched endpoint perturbations control integrated L1 error.
"""
struct EulerCharacteristicCurve
    grid::Vector{Float64}
    EulerCharacteristicCurve(grid) = new(_descriptor_grid(grid))
end
output_size(v::EulerCharacteristicCurve) = length(v.grid)
function (v::EulerCharacteristicCurve)(diagrams::AbstractVector{<:PersistenceDiagram})
    result = zeros(length(v.grid))
    for (q,d) in enumerate(diagrams)
        dimension = get(d.meta,:dim,q-1)
        dimension isa Integer && dimension >= 0 ||
            throw(ArgumentError("homology dimensions must be nonnegative integers"))
        for (j,t) in enumerate(v.grid)
            result[j] += (-1.0)^dimension * count(i -> birth(i) <= t < death(i),d)
        end
    end
    return result
end
(v::EulerCharacteristicCurve)(d::PersistenceDiagram) = v([d])

"""
    PersistenceBlock(x_edges, y_edges; alpha=1.0)
    PersistenceBlock(diagrams; size=(10,10), alpha=1.0)

Vectorized persistence blocks in birth-persistence coordinates. A bar `(b,d)`
contributes the indicator of the square centered at `(b,d-b)` with side
`alpha*(d-b)`. Each feature is the **exact integral** over a grid cell, computed
by rectangle intersection; output is a column-major flattened grid. `0 < alpha <= 2`
keeps support above the diagonal. Infinite and zero-length bars are excluded.
The learned constructor fixes the grid from training data. With a fixed grid,
small matched endpoint changes produce small L1 feature changes; unlike sampled
indicators, cell integration captures short bars and boundary crossings.

Reference: *A Computationally Efficient Framework for Vector Representation of
Persistence Diagrams* (2022), https://jmlr.org/papers/v23/21-1129.html.
"""
struct PersistenceBlock
    x_edges::Vector{Float64}
    y_edges::Vector{Float64}
    alpha::Float64
    function PersistenceBlock(x_edges,y_edges; alpha::Real=1.0)
        0 < alpha <= 2 || throw(ArgumentError("alpha must be in (0,2]"))
        x,y = _descriptor_grid(x_edges),_descriptor_grid(y_edges)
        first(y) >= 0 || throw(ArgumentError("persistence grid must be nonnegative"))
        new(x,y,Float64(alpha))
    end
end
function PersistenceBlock(diagrams; size::Tuple{Int,Int}=(10,10), alpha::Real=1.0)
    all(>(0),size) || throw(ArgumentError("grid size must be positive"))
    points = reduce(vcat,_finite_points.(diagrams);init=Tuple{Float64,Float64}[])
    x,y = if isempty(points)
        (0.0,1.0),(0.0,1.0)
    else
        (minimum(b-alpha*(d-b)/2 for (b,d) in points),
            maximum(b+alpha*(d-b)/2 for (b,d) in points)),
        (0.0,maximum((1+alpha/2)*(d-b) for (b,d) in points))
    end
    return PersistenceBlock(range(x...;length=size[1]+1),
        range(y...;length=size[2]+1);alpha=alpha)
end
output_size(v::PersistenceBlock) = (length(v.x_edges)-1)*(length(v.y_edges)-1)
function (v::PersistenceBlock)(diagram::PersistenceDiagram)
    result = zeros(length(v.x_edges)-1,length(v.y_edges)-1)
    for (b,d) in _finite_points(diagram)
        life = d-b
        h = v.alpha*life/2
        for j in axes(result,2), i in axes(result,1)
            dx = max(0.0,min(v.x_edges[i+1],b+h)-max(v.x_edges[i],b-h))
            dy = max(0.0,min(v.y_edges[j+1],life+h)-max(v.y_edges[j],life-h))
            result[i,j] += dx*dy
        end
    end
    return vec(result)
end

"""
    TropicalCoordinates(; order=5, birth_cap=1.0)

Two finite families of max-plus elementary symmetric coordinates: sums of the
largest k persistences and sums of the largest k values
`persistence + min(birth, birth_cap*persistence)`, for k=1:order.
Missing terms are padded with zero. Requires nonnegative births. Zero-length
bars contribute zero, so adding diagonal points does not change the descriptor.
These symmetric tropical polynomials are permutation invariant and Lipschitz
under bottleneck matching, with bounds `2k` and
`k*(2+max(1,2birth_cap))`. A finite truncation need not distinguish every barcode.

Reference: Kališnik (2019), *Tropical Coordinates on the Space of Persistence
Barcodes*, https://doi.org/10.1007/s10208-018-9379-y.
"""
struct TropicalCoordinates
    order::Int
    birth_cap::Float64
    function TropicalCoordinates(;order::Integer=5,birth_cap::Real=1.0)
        order >= 1 || throw(ArgumentError("order must be positive"))
        isfinite(birth_cap) && birth_cap > 0 || throw(ArgumentError("birth_cap must be positive"))
        new(Int(order),Float64(birth_cap))
    end
end
output_size(v::TropicalCoordinates) = 2v.order
function (v::TropicalCoordinates)(diagram::PersistenceDiagram)
    points = _finite_points(diagram)
    all(p -> p[1] >= 0,points) || throw(ArgumentError("tropical coordinates require nonnegative births"))
    life = sort!([d-b for (b,d) in points];rev=true)
    mixed = sort!([d-b+min(b,v.birth_cap*(d-b)) for (b,d) in points];rev=true)
    return vcat([sum(life[1:min(k,length(life))]) for k in 1:v.order],
        [sum(mixed[1:min(k,length(mixed))]) for k in 1:v.order])
end

"""
    TentTemplate(centers; bandwidth=(0.1,0.1))
    TentTemplate(diagrams; size=(10,10))

Sums of compactly supported rectangular tent functions in birth-persistence
coordinates: `max(0, 1-max(abs(b-cb)/hb,abs(l-cl)/hl))`. Centers must be an n×2
matrix and `cl >= hl > 0`, so all tents vanish on the diagonal. The learned
constructor places centers on a regular grid fitted to training diagrams only.
Infinite intervals are ignored. Fixed templates are Lipschitz for Wasserstein-1
matching, including matches to the diagonal; grid fitting itself is not covered
by that stability statement.

Reference: Perea, Munch & Khasawneh (2019), *Approximating Continuous Functions
on Persistence Diagrams Using Template Functions*, https://arxiv.org/abs/1902.07190.
"""
struct TentTemplate
    centers::Matrix{Float64}
    bandwidth::Tuple{Float64,Float64}
    function TentTemplate(centers::AbstractMatrix;bandwidth=(0.1,0.1))
        size(centers,2) == 2 || throw(DimensionMismatch("centers must be n×2"))
        all(isfinite,centers) || throw(ArgumentError("centers must be finite"))
        h = Float64.(bandwidth)
        length(h)==2 && all(x -> isfinite(x) && x>0,h) ||
            throw(ArgumentError("bandwidth must contain two positive numbers"))
        all(>=(h[2]),centers[:,2]) || throw(ArgumentError("tent support must stay above the diagonal"))
        new(Matrix{Float64}(centers),(h[1],h[2]))
    end
end
function TentTemplate(diagrams;size::Tuple{Int,Int}=(10,10))
    all(>(0),size) || throw(ArgumentError("grid size must be positive"))
    x,y = _descriptor_limits(diagrams)
    hx,hy = (x[2]-x[1])/(2size[1]),(y[2]-y[1])/(2size[2])
    centers = [(x[1]+(2i-1)*hx,y[1]+(2j-1)*hy) for j in 1:size[2] for i in 1:size[1]]
    return TentTemplate(reduce(vcat,[permutedims(collect(c)) for c in centers]);bandwidth=(hx,hy))
end
output_size(v::TentTemplate) = size(v.centers,1)
function (v::TentTemplate)(diagram::PersistenceDiagram)
    points = _finite_points(diagram)
    return [sum((max(0.0,1-max(abs(b-v.centers[j,1])/v.bandwidth[1],
        abs(d-b-v.centers[j,2])/v.bandwidth[2])) for (b,d) in points);init=0.0)
        for j in axes(v.centers,1)]
end

"""
    ComplexPolynomial(; length=10, polynomial_type=:S)

Real and imaginary parts of the first `length` non-leading coefficients of the
monic polynomial whose roots represent finite bars. Output has length 2*length,
with real parts followed by imaginary parts. Types `:R`, `:S`, and `:T` use the
Di Fabio–Ferri root maps also exposed by GUDHI. `:S` and `:T` send diagonal points
to zero. Coefficients are continuous for fixed bounded cardinality, but high
degrees may be ill-conditioned; no global bottleneck-Lipschitz bound is claimed.
Type `:R` does not vanish near the diagonal and is unsuitable when that invariance
is required. Infinite and exactly zero-length intervals are excluded.

Reference: Di Fabio & Ferri (2015), https://doi.org/10.1007/978-3-319-23231-7_27.
"""
struct ComplexPolynomial
    length::Int
    polynomial_type::Symbol
    function ComplexPolynomial(;length::Integer=10,polynomial_type::Symbol=:S)
        length >= 1 || throw(ArgumentError("length must be positive"))
        polynomial_type in (:R,:S,:T) || throw(ArgumentError("polynomial_type must be :R, :S, or :T"))
        new(Int(length),polynomial_type)
    end
end
output_size(v::ComplexPolynomial) = 2v.length
function (v::ComplexPolynomial)(diagram::PersistenceDiagram)
    coefficients = zeros(ComplexF64,v.length+1)
    coefficients[1] = 1
    degree = 0
    for (b,d) in _finite_points(diagram)
        a = hypot(b,d)
        z = if v.polynomial_type == :R
            complex(b,d)
        elseif v.polynomial_type == :S
            a == 0 ? zero(ComplexF64) : complex(b,d)*(d-b)/(sqrt(2)*a)
        else
            (d-b)/2 * complex(cos(a)-sin(a),cos(a)+sin(a))
        end
        degree += 1
        for k in min(degree,v.length):-1:1
            coefficients[k+1] -= z*coefficients[k]
        end
    end
    return vcat(real.(coefficients[2:end]),imag.(coefficients[2:end]))
end

"""
    TopologicalVector(; length=10)

The largest `length` values `min(norm(p_i-p_j, Inf), persistence_i/2,
persistence_j/2)` over unordered finite-bar pairs, sorted descending and
zero-padded. Symmetric capping by **both** persistences ensures permutation and
diagonal-point invariance. A point matched to the diagonal contributes at most
its half-persistence. Empty and one-point diagrams yield zeros.

Reference: Carrière, Oudot & Ovsjanikov (2015), https://doi.org/10.1111/cgf.12692.
"""
struct TopologicalVector
    length::Int
    function TopologicalVector(;length::Integer=10)
        length >= 1 || throw(ArgumentError("length must be positive"))
        new(Int(length))
    end
end
output_size(v::TopologicalVector) = v.length
function (v::TopologicalVector)(diagram::PersistenceDiagram)
    points = _finite_points(diagram)
    distances = [min(max(abs(a-c),abs(b-d)),(b-a)/2,(d-c)/2)
        for (i,(a,b)) in enumerate(points) for (c,d) in points[i+1:end]]
    sort!(distances;rev=true)
    result = zeros(v.length)
    count = min(v.length,length(distances))
    result[1:count] .= distances[1:count]
    return result
end

"""
    Atol(diagrams; n_centers=8, rng=MersenneTwister(42), weighting=:cloud, maxiter=100)
    Atol(centers::AbstractMatrix, radii; weighting=:cloud)

ATOL measure vectorization with Gaussian contrast in birth-death coordinates:
each coordinate is `sum(weight_i * exp(-norm(p_i-center)^2/radius^2))`.
The fitted constructor uses seeded weighted Lloyd quantization on training data.
Radii are half nearest-center distances; a single center uses half the training
diameter (or 1 for a singleton). Unused centers are inactive, giving fixed output
size even for empty or small training sets. `weighting=:cloud` gives unit masses;
`:probability` normalizes each nonempty diagram's mass to one.
Codebooks must be fitted within each training fold. The fixed smooth map is stable
for measures of bounded mass, but does not automatically inherit bottleneck
stability with diagonal matching. Infinite/zero-length intervals are excluded.

Reference: Royer et al. (2021), *ATOL: Measure Vectorization for Automatic
Topologically-Oriented Learning*, https://proceedings.mlr.press/v130/royer21a.html.
"""
struct Atol
    centers::Matrix{Float64}
    radii::Vector{Float64}
    active::BitVector
    weighting::Symbol
    function Atol(centers::AbstractMatrix,radii;weighting::Symbol=:cloud,active=trues(size(centers,1)))
        size(centers,2)==2 || throw(DimensionMismatch("centers must be n×2"))
        length(radii)==length(active)==size(centers,1) || throw(DimensionMismatch("radii and centers must agree"))
        all(isfinite,centers) && all(r -> isfinite(r) && r>0,radii) ||
            throw(ArgumentError("centers and positive radii must be finite"))
        weighting in (:cloud,:probability) || throw(ArgumentError("unknown weighting"))
        new(Matrix{Float64}(centers),Float64.(radii),BitVector(active),weighting)
    end
end
function Atol(diagrams;n_centers::Integer=8,rng::AbstractRNG=MersenneTwister(42),
        weighting::Symbol=:cloud,maxiter::Integer=100)
    n_centers>0 && maxiter>0 || throw(ArgumentError("n_centers and maxiter must be positive"))
    weighting in (:cloud,:probability) || throw(ArgumentError("unknown weighting"))
    clouds = _finite_points.(diagrams)
    points = reduce(vcat,clouds;init=Tuple{Float64,Float64}[])
    weights = reduce(vcat,[fill(weighting==:cloud ? 1.0 : 1/max(1,length(c)),length(c))
        for c in clouds];init=Float64[])
    centers = zeros(n_centers,2)
    active = falses(n_centers)
    radii = ones(n_centers)
    isempty(points) && return Atol(centers,radii;weighting=weighting,active=active)
    unique_points = sort!(unique(points))
    k = min(n_centers,length(unique_points))
    selected = [rand(rng,unique_points)]
    for _ in 2:k
        distances = [minimum(sum(abs2,p.-c) for c in selected) for p in unique_points]
        push!(selected,unique_points[argmax(distances)])
    end
    for j in 1:k
        centers[j,:] .= selected[j]
    end
    active[1:k] .= true
    for _ in 1:maxiter
        labels = [argmin([sum(abs2,p.-Tuple(centers[j,:])) for j in 1:k]) for p in points]
        previous = copy(centers)
        for j in 1:k
            ids = findall(==(j),labels)
            isempty(ids) && continue
            mass = sum(weights[ids])
            centers[j,1] = sum(weights[i]*points[i][1] for i in ids)/mass
            centers[j,2] = sum(weights[i]*points[i][2] for i in ids)/mass
        end
        maximum(abs,centers-previous)<1e-10 && break
    end
    if k==1
        diameter = maximum((sqrt(sum(abs2,p.-q)) for p in points for q in points);init=0.0)
        radii[1] = diameter>0 ? diameter/2 : 1.0
    else
        for j in 1:k
            distances = [norm(centers[j,:]-centers[l,:])/2 for l in 1:k if l!=j]
            radii[j] = max(minimum(distances),eps(Float64))
        end
    end
    return Atol(centers,radii;weighting=weighting,active=active)
end
output_size(v::Atol) = size(v.centers,1)
function (v::Atol)(diagram::PersistenceDiagram)
    points = _finite_points(diagram)
    mass = v.weighting==:cloud ? 1.0 : 1/max(1,length(points))
    return [v.active[j] ? mass*sum((exp(-((b-v.centers[j,1])^2+
        (d-v.centers[j,2])^2)/v.radii[j]^2) for (b,d) in points);init=0.0) : 0.0
        for j in axes(v.centers,1)]
end
