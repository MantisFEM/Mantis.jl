module TimeIntegratorsTests

using Test

@testset verbose=true "Basic Tests" begin
    include("BasicTests.jl")
end
@testset verbose=true "Advection-Based Tests" begin
    include("AdvectionTests.jl")
end
@testset verbose=true "Convergence Tests" begin
    include("ConvergenceTests.jl")
end
@testset verbose=true "Amplification Factors" begin
    include("StabilityTests.jl")
end
@testset verbose=true "Time-dependent Convergence Tests" begin
    include("TimeConvergenceTests.jl")
end

end
