module FiniteElementSpaceTests

using Test

@testset verbose = true "UnivariateSplines" begin
    include("UnivariateSplines/runtests.jl")
end

end
