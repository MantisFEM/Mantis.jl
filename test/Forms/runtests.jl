import Mantis

module FormTests

using Test

@testset "ExteriorDerivatives" verbose=true begin
    include("ExteriorDerivativeTests.jl")
end
@testset "FormEvaluations" verbose=true begin
    include("FormEvaluationTests.jl")
end
@testset "Hodges" verbose=true begin
    include("HodgeTests.jl")
end
@testset "Wedge" verbose=true begin
    include("WedgeTests.jl")
end
@testset "Integral" verbose=true begin
    include("IntegralTests.jl")
end
@testset "Algebraic" verbose=true begin
    include("FormOperators/Algebraic.jl")
end

end
