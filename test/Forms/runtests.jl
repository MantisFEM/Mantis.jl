import Mantis

module FormTests

using Test

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
