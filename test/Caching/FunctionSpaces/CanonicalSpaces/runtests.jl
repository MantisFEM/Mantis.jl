module CanonicalSpacesTests

using Test

@testset verbose = true "Abstract" begin
    include("AbstractTests.jl")
end

@testset verbose = true "Bernstein" begin
    include("BernsteinPolynomialsTests.jl")
end

end
