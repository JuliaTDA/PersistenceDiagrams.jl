using TDAPersistenceDiagrams, Test

function compare_gudhi()
    points = [parse.(Float64,split(line,',')) for line in
        readlines(joinpath(@__DIR__,"gudhi_points.csv"))[2:end]]
    expected = [parse.(Float64,split(line,',')) for line in
        readlines(joinpath(@__DIR__,"gudhi_expected.csv"))[2:end]]
    errors = zeros(3)
    @testset "pinned GUDHI reference" begin
        for row in expected
            case,n,m,slices,sw,swk,fk = row
            diagrams = [PersistenceDiagram([(p[3],p[4]) for p in points
                if p[1]==case && p[2]==side]) for side in (0,1)]
            @test length.(diagrams)==[n,m]
            actual = (SlicedWasserstein(;slices=Int(slices))(diagrams...),
                SlicedWassersteinKernel(;slices=Int(slices),bandwidth=0.8)(diagrams...),
                PersistenceFisherKernel(;sigma=0.7,bandwidth=1.2)(diagrams...))
            for (i,(value,reference)) in enumerate(zip(actual,(sw,swk,fk)))
                # acos magnifies roundoff near coincident probability measures.
                tolerance = i==3 ? 2e-8 : 1e-10
                @test value ≈ reference atol=tolerance rtol=tolerance
                errors[i] = max(errors[i],abs(value-reference))
            end
        end
    end
    println("Maximum errors (SW, SW kernel, Fisher kernel): ",errors)
    return errors
end

compare_gudhi()
