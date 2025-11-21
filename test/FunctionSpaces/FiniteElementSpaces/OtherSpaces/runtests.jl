module OtherSpacesTests

using Test

@testset "MultiComponentMultiPatch" begin
    include("MCMPTests.jl")
end
@testset "DirectSumSpaces" begin
    include("DirectSumSpaceTests.jl")
end

end
