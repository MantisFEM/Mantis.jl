module GeometryTests

using Test

@testset "CartesianGeometry" begin
    include("CartesianGeometryTests.jl")
end
@testset "MappedGeometry" begin
    include("MappedGeometryTests.jl")
end
@testset "FEGeometry" begin
    include("FEGeometryTests.jl")
end
@testset "TensorProductGeometry" begin
    include("TensorProductGeometryTests.jl")
end
@testset "Metric" begin
    include("MetricTests.jl")
end
@testset "Inference" begin
    include("GeometryInferenceTests.jl")
end

end
