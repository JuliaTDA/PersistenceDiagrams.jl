@info "build started."
using Documenter
using TDAPersistenceDiagrams

makedocs(;
    sitename="TDAPersistenceDiagrams.jl",
    format=Documenter.HTML(
        # Use clean URLs, unless built as a "local" build;
        ;
        prettyurls=!("local" in ARGS),
        assets=["assets/favicon.ico"],
    ),
    pages=[
        "Home" => "index.md",
        "Basics" => "basics.md",
        "Distances and Matchings" => "distances.md",
        "Kernels" => "kernels.md",
        "Diagram averaging" => "means.md",
        "Vectorization" => "vectorization.md",
        "Additional descriptors" => "descriptors.md",
        "MLJ Models" => "mlj.md",
        "Validation and benchmarks" => "validation.md",
    ],
    doctest=false, # Doctests are run as part of testing -- no need to run them twice.
)

deploydocs(; repo="github.com/JuliaTDA/PersistenceDiagrams.jl.git")
