module JETTests

using Test

@testset "TensorProducts" begin
    include("TensorProductsInferenceTests.jl")
end
@testset "Geometry" begin
    include("GeometryInferenceTests.jl")
end
@testset "FESpaces" begin
    include("FESpacesInferenceTests.jl")
end
@testset verbose=true "Forms" begin
    @testset "CoDifferential" begin
        include("Forms/CoDifferentialInferenceTests.jl")
    end
    @testset "ExteriorDerivative" begin
        include("Forms/ExteriorDerivativeInferenceTests.jl")
    end
    @testset "Integral" begin
        include("Forms/IntegralInferenceTests.jl")
    end
    @testset "Hodge" begin
        include("Forms/HodgeInferenceTests.jl")
    end
    @testset "Sharp" begin
        include("Forms/SharpInferenceTests.jl")
    end
    @testset "Wedge" begin
        include("Forms/WedgeInferenceTests.jl")
    end
end
@testset "TimeIntegrators" begin
    include("TimeIntegratorsInferenceTests.jl")
end

end
