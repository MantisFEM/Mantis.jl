module MantisTests

using Test

@testset "Mesh" begin include("Mesh/runtests.jl") end
@testset "CanonicalSpaces" begin include("CanonicalSpaces/runtests.jl") end
@testset "Quadrature" begin include("Quadrature/runtests.jl") end
@testset "FiniteElementSpaces" begin include("FiniteElementSpaces/runtests.jl") end
@testset "Plot" begin include("Plot/runtests.jl") end
@testset "Geometry" begin include("Geometry/runtests.jl") end
# @testset "HierarchicalFiniteElementSpaces" begin include("HierarchicalFiniteElementSpaces/runtests.jl") end

end; nothing