module GeometryTests

using Test

@testset "CartesianGeometry" begin include("CartesianGeometry.jl") end
@testset "MappedGeometry" begin include("MappedGeometry.jl") end
@testset "FEMGeometry" begin include("FEMGeometry.jl") end
@testset "TensorProductGeometry" begin include("TensorProductGeometry.jl") end


end