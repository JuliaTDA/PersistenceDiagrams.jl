using SafeTestsets
using Test

@safetestset "descriptors" begin
    include("descriptors.jl")
end
@safetestset "tuning" begin
    include("tuning.jl")
end

@safetestset "diagrams" begin
    include("diagrams.jl")
end
#@safetestset "matching" begin
#    include("matching.jl")
#end
@safetestset "matching" begin
    include("matching.jl")
end
@safetestset "kernels" begin
    include("kernels.jl")
end
@safetestset "GUDHI numerical reference" begin
    include("../validation/compare_gudhi.jl")
end
@safetestset "Fréchet means" begin
    include("means.jl")
end
@safetestset "persistencecurves" begin
    include("persistencecurves.jl")
end
@safetestset "persistenceimages" begin
    include("persistenceimages.jl")
end
@safetestset "scitypes" begin
    include("scitypes.jl")
end
@safetestset "mlj" begin
    include("mlj.jl")
end
@safetestset "plotsrecipes" begin
    include("plotsrecipes.jl")
end
@safetestset "aqua" begin
    include("aqua.jl")
end
@safetestset "doctests" begin
    include("doctests.jl")
end
