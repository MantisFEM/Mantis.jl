module MantisTests

using Test

@testset "Mesh" begin include("Mesh/runtests.jl") end
@testset "CanonicalSpaces" begin include("CanonicalSpaces/runtests.jl") end
@testset "Quadrature" begin include("Quadrature/runtests.jl") end
@testset "FiniteElementSpaces" begin include("FiniteElementSpaces/runtests.jl") end

end; nothing