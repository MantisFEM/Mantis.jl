module OtherSpacesTests

using Test

@testset "MultiComponentMultiPatch" begin
    include("MCMPTests.jl")
end
@testset "DirectSumSpaces" begin
    include("DirectSumSpaceTests.jl")
end
@testset "RationalFESpaces" begin
    include("RationalFESpacesTests.jl")
end

end
