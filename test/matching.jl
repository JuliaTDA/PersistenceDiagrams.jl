using PersistenceDiagrams
using Test

using PersistenceDiagrams:
    BottleneckGraph,
    _adjacency_matrix,
    _depth_layers,
    _augmenting_paths,
    _augment!,
    _hopcroft_karp!

@testset "_adjacency_matrix" begin
    diag1 = PersistenceDiagram([(1, 2), (1, Inf)])
    diag2 = PersistenceDiagram([(3, 4), (5, 10), (7, Inf)])

    @test _adjacency_matrix(diag1, diag2) == [
        # 1,2 1,∞ 3.5,3.5 7.5,7.5 ∞,∞
        2.0 Inf 0.5 Inf Inf  # 3,4
        8.0 Inf Inf 2.5 Inf  # 5,10
        Inf 6.0 Inf Inf Inf  # 7,∞
        0.5 Inf 0.0 0.0 0.0  # 1.5,1.5
        Inf Inf 0.0 0.0 0.0  # ∞,∞
    ]

    @test _adjacency_matrix(diag1, diag2) == _adjacency_matrix(diag2, diag1)'
end

@testset "Hopcroft-Karp" begin
    adj = [
        1 1 9
        1 9 1
        9 9 2
    ]
    graph = BottleneckGraph(adj, [0, 0, 0], [0, 0, 0], Int[], 3)

    @test _depth_layers(graph, 2) == ([1, 1, 1], 1)
    @test _augmenting_paths(graph, 2) == [[1, 1], [3, 2]]

    graph = BottleneckGraph(adj, [1, 0, 2], [1, 3, 0], Int[], 3)
    @test _depth_layers(graph, 2) == ([1, 2, 3], 3)
    @test _augmenting_paths(graph, 2) == [[2, 1, 1, 2, 3, 3]]
    _augment!(graph, [2, 1, 1, 2, 3, 3])
    @test graph.match_left == [2, 1, 3]
    @test graph.match_right == [2, 1, 3]

    graph = BottleneckGraph(adj, [0, 0, 0], [0, 0, 0], Int[], 3)
    @test _hopcroft_karp!(graph, 2) == ([1 => 2, 2 => 1, 3 => 3], true)
    @test _hopcroft_karp!(graph, 1) == ([1 => 1, 3 => 2], false)
end

@testset "Bottleneck basics" begin
    diag1 = PersistenceDiagram([(1, 2), (5, 8)])
    diag2 = PersistenceDiagram([(1, 2), (3, 4), (5, 10)])

    m = Bottleneck()(diag1, diag2; matching=true)
    @test matching(m) == [(5, 8) => (5, 10)]
    @test matching(m; bottleneck=false) ==
        [(1, 2) => (1, 2), (3.5, 3.5) => (3, 4), (5, 8) => (5, 10)]
    @test weight(m) ≡ 2.0
    @test Bottleneck()(diag1, diag2) ≡ 2.0
    @test Bottleneck()(diag1, diag2) == weight(matching(Bottleneck(), diag2, diag1))

    @test weight(Bottleneck(), diag1, diag1) ≡ 0.0
end

@testset "Wasserstein basics" begin
    diag1 = PersistenceDiagram([(1, 2), (5, 8)])
    diag2 = PersistenceDiagram([(1, 2), (3, 4), (5, 10)])

    m = Wasserstein()(diag1, diag2; matching=true)
    @test matching(m) == [(1, 2) => (1, 2), (3.5, 3.5) => (3, 4), (5, 8) => (5, 10)]
    @test weight(m) ≡ 2.5
    @test Wasserstein()(diag1, diag2) ≡ 2.5
    @test weight(matching(Wasserstein(2), diag1, diag2)) ≡ √(0.25 + 4)
    for i in 1:3
        @test Wasserstein(i)(diag1, diag2) ≡ Wasserstein(i)(diag2, diag1)
    end

    @test weight(Wasserstein(), diag1, diag1) ≡ 0.0
end

@testset "Empty diagrams" begin
    diag = PersistenceDiagram([])

    @test Wasserstein()(diag,diag; matching = false) ≡ 0.0
    @test Bottleneck()(diag,diag; matching = false) ≡ 0.0

end

@testset "Infinite intervals" begin
    diag1 = PersistenceDiagram([(1, 2), (5, 8), (1, Inf)])
    diag2 = PersistenceDiagram([(1, 2), (3, 4), (5, 10)])
    diag3 = PersistenceDiagram([(1, 2), (3, 4), (5, 10), (1, Inf)])
    diag4 = PersistenceDiagram([(1, Inf)])
    diag5 = PersistenceDiagram([(2, Inf)])

    for Distance in (Bottleneck(), Wasserstein(), Wasserstein(2))
        @test Distance(diag1, diag2) ≡ Inf
        @test Distance(diag2, diag1) ≡ Inf
        @test weight(Distance(diag1, diag2; matching=true)) ≡ Inf
        @test isempty(matching(Distance, diag1, diag2))
        @test Distance(diag1, diag1) == 0
        @test 0 < Distance(diag1, diag3) < Inf
        @test Distance(diag4, diag5) == 1
    end
end

@testset "Different sizes" begin
    diag1 = PersistenceDiagram(vcat((90, 100), [(i, i + 1) for i in 1:100]))
    diag2 = PersistenceDiagram([(100, 110)])

    @test Bottleneck()(diag1, diag2) == 5.0
    @test Bottleneck()(diag2, diag1) == 5.0
    @test Wasserstein()(diag1, diag2) == 60
    @test Wasserstein()(diag2, diag1) == 60
    @test Wasserstein(2)(diag1, diag2) == √75
    @test Wasserstein(2)(diag2, diag1) == √75
end

@testset "Collections" begin
    diags1 = [
        PersistenceDiagram([(1, 2), (5, 8), (1, Inf)]),
        PersistenceDiagram(vcat((90, 100), [(i, i + 1) for i in 1:100])),
    ]
    diags2 = [
        PersistenceDiagram([(1, 2), (3, 4), (5, 10), (1, Inf)]),
        PersistenceDiagram([(100, 110)]),
    ]

    @test Bottleneck()(diags1, diags2) == 5.0
    @test Bottleneck()(diags1, diags2; matching=true)[2] isa PersistenceDiagrams.Matching
    @test Wasserstein()(diags1, diags2) == 62.5
    @test weight(Wasserstein(3)(diags1, diags2; matching=true)[1]) ==
        Wasserstein(3)(diags1[1], diags2[1])

    @test_throws ArgumentError Bottleneck()(diags1[1:1], diags2)
    @test_throws ArgumentError Wasserstein()(diags1, diags2[2:2])
end

@testset "SlicedWasserstein" begin
    diag1 = PersistenceDiagram([(1, 2), (5, 8)])
    diag2 = PersistenceDiagram([(1, 2), (3, 4), (5, 10)])
    empty = PersistenceDiagram(PersistenceInterval[])

    @testset "constructor" begin
        @test SlicedWasserstein() isa PersistenceDiagrams.MatchingDistance
        @test SlicedWasserstein().slices == 50
        @test SlicedWasserstein(; slices=10).slices == 10
        @test_throws ArgumentError SlicedWasserstein(; slices=0)
        @test_throws ArgumentError SlicedWasserstein(; slices=-1)
    end

    @testset "self distance is zero" begin
        @test SlicedWasserstein()(diag1, diag1) == 0.0
        @test SlicedWasserstein()(diag2, diag2) == 0.0
        @test SlicedWasserstein(; slices=13)(diag2, diag2) == 0.0
    end

    @testset "symmetry" begin
        @test SlicedWasserstein()(diag1, diag2) == SlicedWasserstein()(diag2, diag1)
        @test SlicedWasserstein(; slices=7)(diag1, diag2) ==
            SlicedWasserstein(; slices=7)(diag2, diag1)
    end

    @testset "non-negativity and positivity for distinct diagrams" begin
        @test SlicedWasserstein()(diag1, diag2) > 0
    end

    @testset "hand-computable example" begin
        # A single point (0, 2) versus an empty diagram. Its diagonal projection is the
        # midlife point (1, 1). For a direction θ, the point projects to 2sinθ and the
        # diagonal point to cosθ + sinθ, so the per-slice L1 distance is |sinθ - cosθ|.
        # With `slices=2` the directions are θ = -π/2 and θ = 0, giving |−1−0| = 1 and
        # |0−1| = 1. Averaging (sum/slices) and applying the 1/π factor gives 2/2/π = 1/π.
        point = PersistenceDiagram([(0.0, 2.0)])
        @test SlicedWasserstein(; slices=2)(point, empty) ≈ 1 / π

        # Two equal single-point diagrams are at distance zero regardless of slices.
        @test SlicedWasserstein(; slices=2)(point, point) == 0.0
    end

    @testset "qualitatively tracks Wasserstein-1" begin
        # Moving one point further away should only increase the sliced distance.
        near = PersistenceDiagram([(1, 2), (5, 8.1)])
        far = PersistenceDiagram([(1, 2), (5, 20.0)])
        @test SlicedWasserstein()(diag1, near) < SlicedWasserstein()(diag1, far)
        # Closer to itself than to a different diagram.
        @test SlicedWasserstein()(diag1, near) < SlicedWasserstein()(diag1, diag2) ||
            SlicedWasserstein()(diag1, near) < SlicedWasserstein()(near, diag2)
    end

    @testset "empty diagrams" begin
        @test SlicedWasserstein()(empty, empty) == 0.0
        @test SlicedWasserstein()(empty, diag2) > 0
        # Distance between empty and non-empty is symmetric.
        @test SlicedWasserstein()(empty, diag2) == SlicedWasserstein()(diag2, empty)
    end

    @testset "infinite intervals are ignored" begin
        inf1 = PersistenceDiagram([(1, 2), (5, 8), (1, Inf)])
        @test SlicedWasserstein()(inf1, diag2) == SlicedWasserstein()(diag1, diag2)
        only_inf = PersistenceDiagram([(1, Inf)])
        @test SlicedWasserstein()(only_inf, empty) == 0.0
    end

    @testset "determinism and stability with more slices" begin
        # Evenly spaced (non-random) angles make the result reproducible.
        @test SlicedWasserstein(; slices=20)(diag1, diag2) ==
            SlicedWasserstein(; slices=20)(diag1, diag2)
        # The approximation stabilizes as the number of slices grows.
        v50 = SlicedWasserstein(; slices=50)(diag1, diag2)
        v100 = SlicedWasserstein(; slices=100)(diag1, diag2)
        v200 = SlicedWasserstein(; slices=200)(diag1, diag2)
        @test abs(v100 - v200) < abs(v50 - v200)
        @test abs(v100 - v200) < 1e-2
    end

    @testset "no matching" begin
        @test_throws ArgumentError SlicedWasserstein()(diag1, diag2; matching=true)
        @test_throws ArgumentError matching(SlicedWasserstein(), diag1, diag2)
    end

    @testset "collections sum over pairs" begin
        diags1 = [diag1, diag2]
        diags2 = [diag2, diag1]
        @test SlicedWasserstein()(diags1, diags2) ≈
            SlicedWasserstein()(diag1, diag2) + SlicedWasserstein()(diag2, diag1)
        @test_throws ArgumentError SlicedWasserstein()(diags1, diags2[1:1])
    end
end
