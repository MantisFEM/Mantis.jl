module FormOperatorTests

using Test

@testset "Algebraic.jl" verbose = true begin
    include("Algebraic.jl")
end
@testset "Codifferential" verbose = true begin
    include("CodifferentialTests.jl")
end
@testset "Hodge" verbose = true begin
    include("HodgeTests.jl")
end
@testset "Wedge" verbose = true begin
    include("WedgeTests.jl")
end
@testset "ExteriorDerivative" verbose = true begin
    include("ExteriorDerivativeTests.jl")
end
@testset "Integral" verbose = true begin
    include("IntegralTests.jl")
end

end
