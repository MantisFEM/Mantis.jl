module GeometryTests

using Test

@testset verbose = true "Abstract" begin
    include("AbstractTests.jl")
end

@testset verbose = true "Cartesian" begin
    include("CartesianTests.jl")
end

end
