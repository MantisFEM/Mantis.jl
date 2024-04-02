module MantisTests

using Test

@testset "Mesh" begin include("Mesh/runtests.jl") end
@testset "Polynomials" begin include("Polynomials/runtests.jl") end
@testset "Quadrature" begin include("Quadrature/runtests.jl") end
@testset "FunctionSpaces" begin include("FunctionSpaces/runtests.jl") end
@testset "Hierarchy" begin include("Hierarchy/runtests.jl") end

end; nothing