module ElementSpacesTests

using Test

@testset "LagrangePolynomials" begin include("LagrangePolynomialsTests.jl") end

@testset "BernsteinPolynomial" begin include("BernsteinPolynomialTest.jl") end

end