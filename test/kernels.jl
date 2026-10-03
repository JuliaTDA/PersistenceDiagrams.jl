using TDAPersistenceDiagrams
using Test
using LinearAlgebra

using TDAPersistenceDiagrams: AbstractPersistenceKernel

@testset "SlicedWassersteinKernel" begin
    diag1 = PersistenceDiagram([(1, 2), (5, 8)])
    diag2 = PersistenceDiagram([(1, 2), (3, 4), (5, 10)])
    empty = PersistenceDiagram(PersistenceInterval[])

    @testset "constructor" begin
        @test SlicedWassersteinKernel() isa AbstractPersistenceKernel
        @test SlicedWassersteinKernel().bandwidth == 1.0
        @test SlicedWassersteinKernel().distance.slices == 50
        @test SlicedWassersteinKernel(; slices=10, bandwidth=2.0).distance.slices == 10
        @test SlicedWassersteinKernel(; slices=10, bandwidth=2.0).bandwidth == 2.0
        @test_throws ArgumentError SlicedWassersteinKernel(; bandwidth=0)
        @test_throws ArgumentError SlicedWassersteinKernel(; bandwidth=-1)
        @test_throws ArgumentError SlicedWassersteinKernel(; slices=0)
    end

    @testset "self kernel is one" begin
        @test SlicedWassersteinKernel()(diag1, diag1) == 1.0
        @test SlicedWassersteinKernel()(diag2, diag2) == 1.0
        @test SlicedWassersteinKernel(; bandwidth=3.0)(diag1, diag1) == 1.0
    end

    @testset "symmetry" begin
        @test SlicedWassersteinKernel()(diag1, diag2) ==
            SlicedWassersteinKernel()(diag2, diag1)
        @test SlicedWassersteinKernel(; slices=7, bandwidth=0.5)(diag1, diag2) ==
            SlicedWassersteinKernel(; slices=7, bandwidth=0.5)(diag2, diag1)
    end

    @testset "bounds" begin
        k = SlicedWassersteinKernel()(diag1, diag2)
        @test 0 < k <= 1
    end

    @testset "monotonicity" begin
        # diag1 is closer (in sliced Wasserstein distance) to `near` than to diag2.
        near = PersistenceDiagram([(1, 2), (5, 8.5)])
        @test SlicedWasserstein()(diag1, near) < SlicedWasserstein()(diag1, diag2)
        @test SlicedWassersteinKernel()(diag1, near) >
            SlicedWassersteinKernel()(diag1, diag2)
    end

    @testset "bandwidth effect" begin
        # For distinct diagrams, shrinking the bandwidth lowers the kernel value.
        small = SlicedWassersteinKernel(; bandwidth=0.3)(diag1, diag2)
        large = SlicedWassersteinKernel(; bandwidth=3.0)(diag1, diag2)
        @test small < large
    end

    @testset "empty diagrams" begin
        @test SlicedWassersteinKernel()(empty, empty) == 1.0
        @test 0 < SlicedWassersteinKernel()(empty, diag2) < 1
        @test SlicedWassersteinKernel()(empty, diag2) ==
            SlicedWassersteinKernel()(diag2, empty)
    end
end

@testset "PersistenceFisherKernel" begin
    diag1 = PersistenceDiagram([(1, 2), (5, 8)])
    diag2 = PersistenceDiagram([(1, 2), (3, 4), (5, 10)])
    empty = PersistenceDiagram(PersistenceInterval[])

    @testset "constructor" begin
        @test PersistenceFisherKernel() isa AbstractPersistenceKernel
        @test PersistenceFisherKernel().bandwidth == 1.0
        @test PersistenceFisherKernel().sigma == 1.0
        @test PersistenceFisherKernel(; bandwidth=2.0, sigma=0.5).bandwidth == 2.0
        @test PersistenceFisherKernel(; bandwidth=2.0, sigma=0.5).sigma == 0.5
        @test_throws ArgumentError PersistenceFisherKernel(; bandwidth=0)
        @test_throws ArgumentError PersistenceFisherKernel(; bandwidth=-1)
        @test_throws ArgumentError PersistenceFisherKernel(; sigma=0)
        @test_throws ArgumentError PersistenceFisherKernel(; sigma=-1)
    end

    @testset "self kernel is one" begin
        @test PersistenceFisherKernel()(diag1, diag1) == 1.0
        @test PersistenceFisherKernel()(diag2, diag2) == 1.0
        @test PersistenceFisherKernel(; sigma=2.0)(diag1, diag1) == 1.0
    end

    @testset "symmetry" begin
        @test PersistenceFisherKernel()(diag1, diag2) ==
            PersistenceFisherKernel()(diag2, diag1)
        @test PersistenceFisherKernel(; bandwidth=0.5, sigma=2.0)(diag1, diag2) ==
            PersistenceFisherKernel(; bandwidth=0.5, sigma=2.0)(diag2, diag1)
    end

    @testset "bounds" begin
        k = PersistenceFisherKernel()(diag1, diag2)
        @test 0 < k <= 1
    end

    @testset "monotonicity" begin
        near = PersistenceDiagram([(1, 2), (5, 8.5)])
        far = PersistenceDiagram([(1, 2), (5, 30.0)])
        @test PersistenceFisherKernel()(diag1, near) >
            PersistenceFisherKernel()(diag1, far)
    end

    @testset "empty diagrams" begin
        # Two empty diagrams: d_FIM = 0 by convention, so k = 1.
        @test PersistenceFisherKernel()(empty, empty) == 1.0
        # Empty vs non-empty: handled via the augmentation (only diagonal projections).
        @test 0 < PersistenceFisherKernel()(empty, diag2) <= 1
        @test PersistenceFisherKernel()(empty, diag2) ==
            PersistenceFisherKernel()(diag2, empty)
    end

    @testset "hand-check: identical one-point diagrams" begin
        # Two identical single-point diagrams must give exactly k = 1 (d_FIM = 0).
        point = PersistenceDiagram([(0.0, 2.0)])
        @test PersistenceFisherKernel()(point, point) == 1.0

        # The clamp prevents `acos` domain errors for near-identical diagrams: the inner
        # product can slightly exceed 1 in floating point.
        almost = PersistenceDiagram([(0.0, 2.0 + 1e-12)])
        v = PersistenceFisherKernel()(point, almost)
        @test 0 < v <= 1
        @test !isnan(v)
        @test v ≈ 1.0 atol = 1e-6
    end

    @testset "infinite intervals are ignored" begin
        inf1 = PersistenceDiagram([(1, 2), (5, 8), (1, Inf)])
        @test PersistenceFisherKernel()(inf1, diag2) ==
            PersistenceFisherKernel()(diag1, diag2)
    end
end

@testset "kernel_matrix" begin
    diagrams = [
        PersistenceDiagram([(1.0, 2.0)]),
        PersistenceDiagram([(1.0, 2.0), (3.0, 4.0)]),
        PersistenceDiagram([(0.0, 5.0)]),
    ]

    @testset "SlicedWassersteinKernel" begin
        k = SlicedWassersteinKernel()
        G = kernel_matrix(k, diagrams)
        @test G isa Symmetric
        @test size(G) == (3, 3)
        @test G == G'
        @test all(diag(G) .== 1)         # unit diagonal for the SW kernel
        @test all(0 .< G .<= 1)
        # Entries agree with direct evaluation.
        for i in 1:3, j in 1:3
            @test G[i, j] ≈ k(diagrams[i], diagrams[j])
        end
        # PSD: provably PSD kernel (Carrière et al.); eigenvalues are non-negative.
        @test eigmin(Matrix(G)) >= -1e-10
    end

    @testset "PersistenceFisherKernel" begin
        k = PersistenceFisherKernel()
        G = kernel_matrix(k, diagrams)
        @test G isa Symmetric
        @test size(G) == (3, 3)
        @test G == G'
        @test all(diag(G) .== 1)
        @test all(0 .< G .<= 1)
    end

    @testset "symmetric=false matches symmetric=true" begin
        k = SlicedWassersteinKernel()
        Gs = kernel_matrix(k, diagrams; symmetric=true)
        Gn = kernel_matrix(k, diagrams; symmetric=false)
        @test Gn isa Matrix{Float64}
        @test Matrix(Gs) == Gn
    end
end
