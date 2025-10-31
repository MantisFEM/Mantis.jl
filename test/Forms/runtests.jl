import Mantis

module FormTests

using Test

@testset "ExteriorDerivatives" begin
    include("ExteriorDerivativeTests.jl")
end
@testset "FormEvaluations" begin
    include("FormEvaluationTests.jl")
end
@testset "Hodges" begin
    include("HodgeTests.jl")
end
@testset "Wedge" begin
    include("WedgeTests.jl")
end
@testset "Integral" begin
    include("IntegralTests.jl")
end
@testset "Algebraic" begin
    include("FormOperators/Algebraic.jl")
end

end
