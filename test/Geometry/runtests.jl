module GeometryTests

using Test

@testset "CartesianGeometry" verbose=true begin
    include("CartesianGeometryTests.jl")
end
@testset "MappedGeometry" verbose=true begin
    include("MappedGeometryTests.jl")
end
# @testset "DiscreteGeometry" verbose=true begin
#     include("DiscreteGeometryTests.jl")
# end
@testset "TensorProductGeometry" verbose=true begin
    include("TensorProductGeometryTests.jl")
end
@testset "UnstructuredGeometry" verbose=true begin
    include("UnstructuredGeometryTests.jl")
end
@testset "HierarchicalGeometry" verbose=true begin
    include("HierarchicalGeometryTests.jl")
end
@testset "SkeletonGeometry" verbose=true begin
    include("SkeletonGeometryTests.jl")
end
@testset "Metric" verbose=true begin
    include("MetricTests.jl")
end
@testset "ErrorBehaviour" verbose=true begin
    include("GeometryErrorsTests.jl")
end
@testset "EvaluationMask" verbose=true begin
    include("EvaluationMask/runtests.jl")
end

end
