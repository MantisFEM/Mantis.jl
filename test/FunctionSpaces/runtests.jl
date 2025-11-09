module FunctionSpacesTests

using Test

@testset verbose = true "CanonicalSpaces" begin
    include("CanonicalSpaces/runtests.jl")
end

@testset verbose = true "FiniteElementSpaces" begin
    include("FiniteElementSpaces/runtests.jl")
end

@testset verbose = true "AdaptiveRefinement" begin
    include("AdaptiveRefinement/runtests.jl")
end

end
