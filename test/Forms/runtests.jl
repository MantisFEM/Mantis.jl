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
@testset "ComponentMixing" verbose = true begin
    include("ComponentMixingTests.jl")
end
@testset "FormsHelpers" verbose = true begin
    include("FormsHelpersTests.jl")
end

end
