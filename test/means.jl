using TDAPersistenceDiagrams
using Test

empty_diagram = PersistenceDiagram(PersistenceInterval[])
point(b, d) = PersistenceDiagram([(Float64(b), Float64(d))])

@testset "Fréchet means with squared Euclidean Wasserstein cost" begin
    @test Wasserstein(2, 2)(empty_diagram, point(0, 2)) ≈ sqrt(2)
    @test Wasserstein(2, 2)(point(0, 2), empty_diagram) ≈ sqrt(2)
    @test Bottleneck()(empty_diagram, point(0, 2)) == 1
    @test Bottleneck()(point(0, 2), empty_diagram) == 1
    r = frechet_mean([point(0, 2), point(0, 4)])
    @test r.converged
    @test length(r.diagram) == 1
    @test birth(r.diagram[1]) ≈ 0
    @test death(r.diagram[1]) ≈ 3
    @test r.objective ≈ 1
    @test all(diff(r.history) .<= 1e-10)
    @test r.objective ≈ frechet_variance([point(0, 2), point(0, 4)]; center=r.diagram)
    @test diagram_mean([point(0, 2), point(0, 4)]) == r.diagram

    # Diagonal terms affect the normal coordinate but not the tangent coordinate.
    r = frechet_mean([point(0, 2), empty_diagram])
    @test (birth(r.diagram[1]), death(r.diagram[1])) == (0.5, 1.5)
    @test r.objective ≈ 0.5
    @test frechet_mean([point(0, 2), empty_diagram]; init=empty_diagram).diagram == r.diagram
    r = frechet_mean([point(0, 2), empty_diagram]; weights=[3, 1])
    @test (birth(r.diagram[1]), death(r.diagram[1])) == (0.25, 1.75)
    @test r.objective ≈ 0.375

    # The optimal two-diagram matching goes through the diagonal, creating two
    # mean points rather than the (much worse) coordinate-wise average.
    r = frechet_mean([point(0, 2), point(10, 12)])
    @test length(r.diagram) == 2
    @test [(birth(x), death(x)) for x in r.diagram] == [(0.5, 1.5), (10.5, 11.5)]
    @test r.objective ≈ 1

    a = PersistenceDiagram([(0.0, 2.0), (5.0, 8.0)]; dim=1)
    @test frechet_mean([a, a, a]).diagram == a
    @test length(a) == 2 # inputs are not mutated
    @test isempty(frechet_mean([empty_diagram, empty_diagram]).diagram)
    @test frechet_mean([a, empty_diagram]; weights=[1, 0]).diagram == a
    @test frechet_mean([point(0, 2), point(0, 4)]; weights=[1e308, 1e308]).objective ≈ 1

    e1 = PersistenceDiagram([(0.0, Inf), (10.0, Inf)]; dim=0)
    e2 = PersistenceDiagram([(12.0, Inf), (2.0, Inf)]; dim=0)
    r = frechet_mean([e1, e2])
    @test birth.(r.diagram) == [1, 11]
    @test all(!isfinite, r.diagram)
    @test r.objective ≈ 2
    @test dim(r.diagram) == 0
    @test r.objective ≈ frechet_variance([e1, e2]; center=r.diagram)
    @test frechet_variance([e1]; center=empty_diagram) == Inf

    @test_throws ArgumentError frechet_mean(PersistenceDiagram[])
    @test_throws ArgumentError frechet_mean([a]; weights=[0])
    @test_throws ArgumentError frechet_mean([a]; weights=[NaN])
    @test_throws DimensionMismatch frechet_mean([a]; weights=[1, 2])
    @test_throws ArgumentError frechet_mean([e1, empty_diagram])
    @test_throws ArgumentError frechet_mean([point(2, 1)])
    @test_throws ArgumentError frechet_mean([point(NaN, 2)])
    @test_throws ArgumentError frechet_mean([a, PersistenceDiagram([(0, 2)]; dim=0)])
    @test_throws ArgumentError frechet_mean([a]; max_iterations=0)
    @test_throws ArgumentError frechet_mean([a]; atol=-1)

    # Independent closed-form corpus: three singleton diagrams all share the
    # same middle coordinate. Their only global optimum is one averaged point.
    ds = [point(-h, h) for h in (1.0, 2.0, 4.0)]
    r = frechet_mean(ds)
    @test r.converged
    @test length(r.diagram) == 1
    @test birth(r.diagram[1]) ≈ -7 / 3
    @test death(r.diagram[1]) ≈ 7 / 3
    @test r.objective ≈ 28 / 9
    @test all(diff(r.history) .<= 1e-10)
end
