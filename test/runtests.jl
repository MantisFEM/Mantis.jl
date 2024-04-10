module MantisTests

using Test

@testset "Mesh" begin include("Mesh/runtests.jl") end
@testset "ElementSpaces" begin include("ElementSpaces/runtests.jl") end
@testset "Quadrature" begin include("Quadrature/runtests.jl") end
@testset "FiniteElementSpaces" begin include("FiniteElementSpaces/runtests.jl") end
@testset "HierarchicalFiniteElementSpaces" begin include("HierarchicalFiniteElementSpaces/runtests.jl") end

end; nothing