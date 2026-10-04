using BenchmarkTools, TDAPersistenceDiagrams, Random, LinearAlgebra
BLAS.set_num_threads(1)

function benchmark_diagrams(;seed=20261003,samples=20,seconds=1.0,
        output=joinpath(@__DIR__,"baseline.csv"))
    rng = Xoshiro(seed)
    root = dirname(@__DIR__)
    open(output*".metadata","w") do io
        println(io,"cpu = ",Sys.CPU_NAME,"\nos = ",Sys.KERNEL,"\narch = ",Sys.ARCH)
        println(io,"git_revision = ",strip(read(`git -C $root rev-parse HEAD`,String)))
        println(io,"working_tree_dirty = ",!isempty(read(`git -C $root status --porcelain`,String)))
        println(io,"benchmarktools = ",pkgversion(BenchmarkTools))
    end
    diagram(n) = PersistenceDiagram([(b,b+0.01+2rand(rng)) for b in rand(rng,n)])
    rows = NamedTuple[]
    function measure(name,operation,n,slices,diagrams)
        operation() # compilation/warmup excluded
        trial = run(@benchmarkable($operation());samples=samples,seconds=seconds,evals=1)
        result = median(trial)
        push!(rows,(operation=name,n_points=n,slices=slices,n_diagrams=diagrams,
            nanoseconds=result.time,bytes=result.memory,allocations=result.allocs,
            actual_samples=length(trial),seed=seed,julia=string(VERSION),
            threads=Threads.nthreads(),blas_threads=BLAS.get_num_threads(),
            package_version=string(pkgversion(TDAPersistenceDiagrams))))
    end
    for n in (20,100,500),slices in (50,200)
        a,b = diagram(n),diagram(n)
        sw = SlicedWasserstein(;slices=slices)
        kernel = SlicedWassersteinKernel(;slices=slices)
        measure("sliced_wasserstein",()->sw(a,b),n,slices,2)
        measure("sw_kernel",()->kernel(a,b),n,slices,2)
    end
    for count in (10,25)
        diagrams = [diagram(30) for _ in 1:count]
        sw,fisher = SlicedWassersteinKernel(),PersistenceFisherKernel()
        measure("sw_gram",()->kernel_matrix(sw,diagrams),30,50,count)
        measure("fisher_gram",()->kernel_matrix(fisher,diagrams),30,0,count)
    end
    open(output,"w") do io
        println(io,join(string.(keys(first(rows))),','))
        for row in rows
            println(io,join(string.(values(row)),','))
        end
    end
    println("Wrote ",length(rows)," benchmark cases to ",output)
    return rows
end

if abspath(PROGRAM_FILE)==@__FILE__
    benchmark_diagrams(;output=isempty(ARGS) ? joinpath(@__DIR__,"baseline.csv") : ARGS[1])
end
