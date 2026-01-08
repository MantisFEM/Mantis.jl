module AssemblersTests

using Test

@testset "k-form-TensorProduct-L2ProjectionTests" begin
    include("k-form-TensorProduct-L2ProjectionTests.jl")
end
@testset "k-form-Polarsplines-L2ProjectionTests" begin
    include("k-form-PolarSplines-L2ProjectionTests.jl")
end
@testset "0-form-TensorProduct-LaplacianTests" begin
    include("0-form-TensorProduct-LaplacianTests.jl")
end
@testset "1-form-TensorProduct-MaxwellEigenvalueTests" begin
    include("1-form-TensorProduct-MaxwellEigenvalueTests.jl")
end
@testset "1-form-Hierarchical-MaxwellEigenvalueTests" begin
    include("1-form-Hierarchical-MaxwellEigenvalueTests.jl")
end
@testset "n-form-TensorProduct-MixedLaplacianTests" begin
    include("n-form-TensorProduct-MixedLaplacianTests.jl")
end

end
