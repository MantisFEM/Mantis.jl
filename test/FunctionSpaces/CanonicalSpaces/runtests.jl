module CanonicalSpacesTests

using Test

@testset "LagrangePolynomials" begin
    include("LagrangePolynomialsTests.jl")
end
@testset "BernsteinPolynomials" begin
    include("BernsteinPolynomialsTests.jl")
end

@testset verbose=true "ECTSpaces" begin
    include("ECTSpaces/runtests.jl")
end

end
