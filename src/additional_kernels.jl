"""
    PersistenceScaleSpaceKernel(; sigma=1.0)

Persistence scale-space kernel of Reininghaus et al. (2015). `sigma` is the
positive heat diffusion time, not a Gaussian standard deviation. For points
`x` and `y` above the diagonal, the summand is
`(exp(-||x-y||²/(8sigma)) - exp(-||x-reflect(y)||²/(8sigma))) / (8π*sigma)`.

This is an unnormalized positive semidefinite kernel. Empty diagrams have zero
embedding; diagonal points contribute zero. Essential intervals are ignored,
as in the other diagram kernels. Evaluation costs O(length(left)*length(right)).

Reference: https://arxiv.org/abs/1412.6821.
"""
struct PersistenceScaleSpaceKernel <: AbstractPersistenceKernel
    sigma::Float64
    function PersistenceScaleSpaceKernel(; sigma=1.0)
        isfinite(sigma) && sigma > 0 || throw(ArgumentError("sigma must be finite and positive"))
        return new(Float64(sigma))
    end
end

function (k::PersistenceScaleSpaceKernel)(left, right)
    value = 0.0
    for x in left, y in right
        isfinite(x) && isfinite(y) || continue
        isfinite(birth(x)) && isfinite(birth(y)) && birth(x) <= death(x) && birth(y) <= death(y) ||
            throw(ArgumentError("intervals must have birth <= death"))
        a = ((birth(x) - birth(y))^2 + (death(x) - death(y))^2) / (8k.sigma)
        # The reflected squared distance differs by 2*persistence(x)*persistence(y).
        # expm1 preserves precision for points close to the diagonal.
        delta = persistence(x) * persistence(y) / (4k.sigma)
        value += exp(-a) * (-expm1(-delta))
    end
    return value / (8π * k.sigma)
end

"""
    PersistenceWeightedGaussianKernel(; sigma=1.0, C=1.0, power=3.0,
                                       bandwidth=nothing, weight=nothing)

Persistence weighted Gaussian kernel (Kusano, Fukumizu and Hiraoka, 2016).
The default weight is `atan(C*persistence(x)^power)`, and `sigma` is the
Gaussian standard deviation. With `bandwidth=nothing`, return the RKHS inner
product `sum(weight(x)*weight(y)*exp(-||x-y||²/(2sigma²)))`.
With a positive `bandwidth`, return an outer Gaussian kernel of the squared
RKHS embedding distance. Both forms are positive semidefinite.

A custom `weight` must be callable on a `PersistenceInterval`, return a finite
real value and vanish on the diagonal. Stability guarantees from the paper
require its specified persistence-dependent weight and exponent assumptions;
an arbitrary custom weight does not inherit those guarantees. Essential
intervals are ignored. The linear kernel of an empty diagram is zero; the
outer Gaussian self-kernel is one, including for empty diagrams.

Reference: https://arxiv.org/abs/1601.01741.
"""
struct PersistenceWeightedGaussianKernel{W} <: AbstractPersistenceKernel
    sigma::Float64
    C::Float64
    power::Float64
    bandwidth::Union{Nothing,Float64}
    weight::W
    function PersistenceWeightedGaussianKernel(; sigma=1.0, C=1.0, power=3.0,
                                                bandwidth=nothing, weight=nothing)
        for (name, value) in ((:sigma, sigma), (:C, C), (:power, power))
            isfinite(value) && value > 0 ||
                throw(ArgumentError("$name must be finite and positive"))
        end
        isnothing(bandwidth) || (isfinite(bandwidth) && bandwidth > 0) ||
            throw(ArgumentError("bandwidth must be finite and positive"))
        return new{typeof(weight)}(Float64(sigma), Float64(C), Float64(power),
            isnothing(bandwidth) ? nothing : Float64(bandwidth), weight)
    end
end

function _pw_weight(k::PersistenceWeightedGaussianKernel, x)
    isfinite(birth(x)) && persistence(x) >= 0 ||
        throw(ArgumentError("intervals must have finite birth <= death"))
    w = isnothing(k.weight) ? atan(k.C * persistence(x)^k.power) : k.weight(x)
    w isa Real && isfinite(w) || throw(ArgumentError("weight must return a finite real value"))
    iszero(persistence(x)) && !iszero(w) &&
        throw(ArgumentError("weight must vanish on the diagonal"))
    return Float64(w)
end

function _pw_inner(k::PersistenceWeightedGaussianKernel, left, right)
    l = [(x, _pw_weight(k, x)) for x in left if isfinite(x)]
    r = [(y, _pw_weight(k, y)) for y in right if isfinite(y)]
    value = 0.0
    for (x, wx) in l, (y, wy) in r
        value += wx * wy * exp(-((birth(x) - birth(y))^2 +
                                (death(x) - death(y))^2) / (2k.sigma^2))
    end
    return value
end

function (k::PersistenceWeightedGaussianKernel)(left, right)
    cross = _pw_inner(k, left, right)
    isnothing(k.bandwidth) && return cross
    squared = max(0.0, _pw_inner(k, left, left) + _pw_inner(k, right, right) - 2cross)
    return exp(-squared / (2k.bandwidth^2))
end
