using TDAPersistenceDiagrams
using LinearAlgebra

diagrams = [PersistenceDiagram([(0.0, 2.0), (5.0, 7.0)]; dim=1),
            PersistenceDiagram([(0.1, 2.5), (5.2, 7.1)]; dim=1),
            PersistenceDiagram([(0.2, 2.2)]; dim=1)]

# P3.3: weighted diagram averaging with diagnostics.
mean_result = frechet_mean(diagrams; weights=[1, 2, 1])
@assert mean_result.converged
@assert all(diff(mean_result.history) .<= 1e-9)
println("Mean endpoints: ", [(birth(x), death(x)) for x in mean_result.diagram])
println("Weighted squared W2,2 variance: ", mean_result.objective)

# P3.5: both remaining kernel families, ready for kernel methods.
for kernel in (PersistenceScaleSpaceKernel(; sigma=0.5),
               PersistenceWeightedGaussianKernel(; sigma=1.0),
               PersistenceWeightedGaussianKernel(; sigma=1.0, bandwidth=1.0))
    gram = Matrix(kernel_matrix(kernel, diagrams))
    @assert eigmin(gram) >= -1e-10
    println(typeof(kernel), " Gram matrix:\n", gram)
end
