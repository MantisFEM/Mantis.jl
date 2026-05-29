module FunctionSpaceTests

using Test

@testset verbose = true "Canonical" begin
    include("CanonicalSpaces/runtests.jl")
end
@testset verbose = true "FiniteElementSpaces" begin
    include("FiniteElementSpaces/runtests.jl")
end

end
