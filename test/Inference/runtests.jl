module JETTests

using Test

@testset "TensorProducts" begin include("TensorProductsInferenceTests.jl") end
@testset "Geometry" begin include("GeometryInferenceTests.jl") end
@testset "FESpaces" begin include("FESpacesInferenceTests.jl") end
@testset "TimeIntegrators" begin include("TimeIntegratorsInferenceTests.jl") end

end
