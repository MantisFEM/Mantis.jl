module GeometryTests

using Test

@testset "CartesianGeometry" begin
    include("CartesianGeometryTests.jl")
end
@testset "MappedGeometry" begin
    include("MappedGeometryTests.jl")
end
@testset "DiscreteGeometry" begin
    include("DiscreteGeometryTests.jl")
end
@testset "TensorProductGeometry" begin
    include("TensorProductGeometryTests.jl")
end
@testset "UnstructuredGeometry" begin
    include("UnstructuredGeometryTests.jl")
end

@testset "HierarchicalGeometry" verbose = true begin
    include("HierarchicalGeometryTests.jl")
end

@testset "Metric" begin
    include("MetricTests.jl")
end
@testset "ErrorBehaviour" begin
    include("GeometryErrorsTests.jl")
end
@testset "EvaluationMask" verbose=true begin
    include("EvaluationMask/runtests.jl")
end

end
