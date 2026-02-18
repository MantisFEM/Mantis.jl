module UnivariateSplineTests

using Test

# @testset "KnotInsertion" begin include("KnotInsertionTests.jl") end
@testset "BezierExtraction" begin
    include("BezierExtractionTests.jl")
end
@testset "BSplines" begin
    include("BSplineTests.jl")
end

end
