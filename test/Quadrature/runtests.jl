module QuadratureTests

using Test

@testset "ClenshawCurtisQuadrature" begin
    include("ClenshawCurtisTests.jl")
end
@testset "GaussQuadrature" begin
    include("GaussTests.jl")
end
@testset "NewtonCotesQuadrature" begin
    include("NewtonCotesTests.jl")
end
@testset "TensorProductQuadrature" begin
    include("TensorProductTests.jl")
end

end
