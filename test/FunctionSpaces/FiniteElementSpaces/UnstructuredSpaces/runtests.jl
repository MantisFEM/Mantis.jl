module UnstructuredSpacesTests

using Test

@testset "GTBSplines" begin
    include("GTBSplinesTests.jl")
end
@testset "Polarsplines" begin
    include("PolarSplineTests.jl")
end

end
