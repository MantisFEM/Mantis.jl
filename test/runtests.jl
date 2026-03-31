module MantisTests

using Test

@testset verbose=true "Topology" begin include("Topology/runtests.jl") end
@testset verbose=true "Quadrature" begin include("Quadrature/runtests.jl") end
@testset verbose=true "FunctionSpaces" begin include("FunctionSpaces/runtests.jl") end
@testset verbose=true "Geometry" begin include("Geometry/runtests.jl") end
@testset verbose=true "Forms" begin include("Forms/runtests.jl") end
@testset verbose=true "Assembly" begin include("Assemblers/runtests.jl") end
@testset verbose=true "Plot" begin include("Plot/runtests.jl") end
@testset verbose=true "GeneralHelpers" begin include("GeneralHelpers/runtests.jl") end

end; nothing
