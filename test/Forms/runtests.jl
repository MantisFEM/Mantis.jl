import Mantis

module FormTests

using Test

# First tests the pullbacks, since those are used in all subsequent evaluations. Testing
# them first should catch any failures caused by them.
@testset "Pullback" verbose = true begin
    include("FormOperators/Pullback.jl")
end
@testset "FormEvaluations" verbose = true begin
    include("FormEvaluationTests.jl")
end
@testset "FormOperators" verbose = true begin
    include("FormOperators/runtests.jl")
end
@testset "ConstantFormSpaces" verbose = true begin
    include("ConstantFormSpaceTests.jl")
end

end
