module GeometryTests

using Test

@testset "CartesianGeometry" begin
    include("CartesianGeometryTests.jl")
end
@testset "MappedGeometry" begin
    include("MappedGeometryTests.jl")
end
@testset "FEMGeometry" begin
    include("FEMGeometryTests.jl")
end
@testset "TensorProductGeometry" begin
    include("TensorProductGeometryTests.jl")
end
@testset "HierarchicalGeometry" begin
    include("HierarchicalGeometryTests.jl")
end
@testset "NestedGeometry" begin
    include("NestedGeometryTests.jl")
end
@testset "Metric" begin
    include("MetricTests.jl")
end

end
