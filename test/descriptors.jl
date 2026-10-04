using TDAPersistenceDiagrams, Test, Random, Tables
using MLJBase
using TDAPersistenceDiagrams: output_size
import MLJModelInterface as MMI

@testset "Euler characteristic from Betti numbers" begin
    h0 = PersistenceDiagram([(0.0,Inf),(0.0,1.0)];dim=0)
    h1 = PersistenceDiagram([(0.5,2.0)];dim=1)
    curve = EulerCharacteristicCurve([0.0,0.5,1.0,2.0])
    @test curve([h0,h1]) == [2.0,1.0,0.0,1.0]
    @test curve(h1) == [0.0,-1.0,-1.0,0.0]
    @test curve([h1,h0]) == curve([h0,h1])
    @test_throws ArgumentError EulerCharacteristicCurve([1.0,0.0])
end

@testset "integrated persistence blocks and tent templates" begin
    d = PersistenceDiagram([(1.0,3.0)];dim=1)
    block = PersistenceBlock([0.0,1.0,2.0],[0.0,1.0,2.0,3.0];alpha=1.0)
    @test block(d) == [0.0,0.0,1.0,1.0,1.0,1.0]
    @test sum(block(d)) == 4.0 # exact square area (alpha*life)^2
    shifted = PersistenceDiagram([(1.001,3.001)];dim=1)
    @test sum(abs,block(d)-block(shifted)) <= 0.0041
    small = PersistenceDiagram([(0.0,0.01)];dim=1)
    fine = PersistenceBlock([-1.0,1.0],[0.0,1.0])
    @test sum(fine(small)) ≈ 0.0001
    tent = TentTemplate([1.0 2.0];bandwidth=(0.5,0.5))
    @test tent(d) == [1.0]
    @test tent(PersistenceDiagram([(1.0,2.0)])) == [0.0]
    @test tent(PersistenceDiagram([(1.1,3.1)]))[1] ≈ 0.8
    @test_throws ArgumentError TentTemplate([0.0 0.05];bandwidth=(0.1,0.1))
    @test_throws ArgumentError PersistenceBlock([0,1],[0,1];alpha=0)
    @test_throws ArgumentError PersistenceBlock([0,1],[-1,1])
end

@testset "tropical symmetric coordinates" begin
    d = PersistenceDiagram([(1.0,3.0),(0.5,1.5)];dim=1)
    v = TropicalCoordinates(;order=3)
    @test v(d) == [2.0,3.0,3.0,3.0,4.5,4.5]
    diagonal = PersistenceDiagram([(1.0,3.0),(0.5,1.5),(5.0,5.0)];dim=1)
    @test v(d) == v(diagonal)
    @test v(d) == v(PersistenceDiagram(reverse(d.intervals)))
    perturbed = PersistenceDiagram([(1.01,3.02),(0.49,1.51)])
    @test all(abs.(v(d)[1:3]-v(perturbed)[1:3]) .<= 2 .* (1:3) .* 0.02)
    @test all(abs.(v(d)[4:6]-v(perturbed)[4:6]) .<= 4 .* (1:3) .* 0.02)
    @test_throws ArgumentError v(PersistenceDiagram([(-1.0,2.0)]))
end

@testset "complex polynomials and topological vectors" begin
    d = PersistenceDiagram([(0.0,2.0),(1.0,3.0)];dim=1)
    v = ComplexPolynomial(;length=3,polynomial_type=:R)
    # (z-2i)(z-(1+3i)) = z² -(1+5i)z -6+2i.
    @test v(d) ≈ [-1.0,-6.0,0.0,-5.0,2.0,0.0]
    for kind in (:R,:S,:T)
        polynomial = ComplexPolynomial(;length=3,polynomial_type=kind)
        @test polynomial(d) ≈ polynomial(PersistenceDiagram(reverse(d.intervals)))
    end
    v = TopologicalVector(;length=4)
    @test v(d) == [1.0,0.0,0.0,0.0]
    @test v(PersistenceDiagram([(0.0,2.0)])) == zeros(4)
    unequal = PersistenceDiagram([(0.0,4.0),(2.0,2.2),(1.0,3.0)])
    @test v(unequal) ≈ v(PersistenceDiagram(reverse(unequal.intervals)))
    for descriptor in (ComplexPolynomial(),TopologicalVector(),TropicalCoordinates())
        @test all(iszero,descriptor(PersistenceDiagram(PersistenceInterval[])))
    end
end

@testset "ATOL fixed and trained codebooks" begin
    d = PersistenceDiagram([(0.0,1.0),(2.0,3.0)];dim=1)
    v = Atol([0.0 1.0;2.0 3.0],[1.0,1.0])
    @test v(d) ≈ fill(1+exp(-8),2)
    normalized = Atol(v.centers,v.radii;weighting=:probability)
    @test normalized(d) ≈ v(d)/2
    training = [d,PersistenceDiagram([(0.1,1.1),(2.1,3.1)];dim=1)]
    a = Atol(training;n_centers=2,rng=MersenneTwister(42))
    b = Atol(training;n_centers=2,rng=MersenneTwister(42))
    @test a.centers == b.centers
    @test sort(a.centers[:,1]) ≈ [0.05,2.05]
    @test all(>(0),a.radii)
    @test all(isfinite,a(d))
    small = Atol([d];n_centers=5,rng=MersenneTwister(1))
    @test output_size(small)==5
    @test count(small.active)==2
    @test all(iszero,small(d)[.!small.active])
    empty = Atol([PersistenceDiagram(PersistenceInterval[])];n_centers=3)
    @test empty(d)==zeros(3)
    @test_throws ArgumentError Atol([d];n_centers=0)
    @test_throws ArgumentError Atol([0.0 1.0],[0.0])
end

@testset "all descriptor shapes, finite output and MLJ training isolation" begin
    empty = PersistenceDiagram(PersistenceInterval[];dim=1)
    train = (h0=[PersistenceDiagram([(0.0,Inf),(0.0,1.0)];dim=0),
                 PersistenceDiagram([(0.0,Inf),(0.0,1.1)];dim=0)],
             h1=[PersistenceDiagram([(0.1,1.0)];dim=1),empty])
    test = (h0=[PersistenceDiagram([(0.0,Inf),(0.0,100.0)];dim=0)],
            h1=[PersistenceDiagram([(50.0,100.0)];dim=1)])
    for kind in (:euler,:block,:tropical,:tent,:polynomial,:topological,:atol)
        model = PersistenceDescriptorVectorizer(;kind=kind,resolution=10,size=(3,4),
            order=3,n_centers=3)
        fitted,_,_ = MMI.fit(model,0,train)
        training_output = MMI.transform(model,fitted,train)
        original = deepcopy(fitted)
        output = MMI.transform(model,fitted,test)
        @test length(Tables.rows(training_output))==2
        @test length(Tables.rows(output))==1
        @test length(Tables.columnnames(training_output))==length(Tables.columnnames(output))
        @test all(isfinite,Tables.matrix(output))
        # Transforming held-out outliers must not refit grids or codebooks.
        if kind==:euler
            @test fitted.curve.grid == original.curve.grid
        else
            for ((_,a),(_,b)) in zip(fitted,original)
                for field in fieldnames(typeof(a))
                    @test getfield(a,field)==getfield(b,field)
                end
                @test length(a(empty))==output_size(a)
                @test all(isfinite,a(empty))
            end
        end
    end
end
