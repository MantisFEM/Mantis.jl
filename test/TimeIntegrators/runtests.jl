module TimeIntegratorsTests

using Test

@testset verbose=true "Basics" begin
    include("BasicTests.jl")
end
@testset verbose=true "Advection-Based" begin
    include("AdvectionTests.jl")
end
@testset verbose=true "Convergence" begin
    include("ConvergenceTests.jl")
end
@testset verbose=true "Stability" begin
    include("StabilityTests.jl")
end
@testset verbose=true "Time-dependent Convergence" begin
    include("TimeConvergenceTests.jl")
end

end
