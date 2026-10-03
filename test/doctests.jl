using Documenter
using TDAPersistenceDiagrams
using Test

if VERSION ≥ v"1.11-DEV" || VERSION < v"1.11-DEV"
    @warn "Doctests were set up on Julia v1.11. Skipping."
else
    DocMeta.setdocmeta!(
        TDAPersistenceDiagrams, :DocTestSetup, :(using TDAPersistenceDiagrams); recursive=true
    )
    doctest(TDAPersistenceDiagrams)
end
