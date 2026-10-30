module UnstructuredGeometryTests

using Mantis

using Test
using StaticArrays

include("../TestHelpers.jl")

@testset "1D" verbose=true begin
    @testset "1 patch, 1 element, Cartesian" verbose=true begin
        geometry = Geometry.UnstructuredGeometry((Geometry.CartesianGeometry(([-1, 1],)),))

        # Geometry.jl
        @test Geometry.get_manifold_dim(geometry) == 1
        @test Geometry.get_image_dim(geometry) == 1
        @test Geometry.get_num_patches(geometry) == 1
        @test Geometry.get_topology(geometry) == geometry.topology
        @test Geometry.get_patch_id(geometry, 1) == 1
        @test_throws ArgumentError Geometry.get_patch_id(geometry, 2)
        @test Geometry.get_patch_and_local_element_id(geometry, 1) == (1, 1)
        @test Geometry.get_global_element_id(geometry, 1, 1) == 1
        @test Geometry.get_num_elements(geometry) == 1
        @test Geometry.get_num_elements(geometry, 1) == 1
        @test Geometry.get_num_elements_per_patch(geometry) == (1,)
        @test Geometry.get_element_measure(geometry, 1) == 2.0
        @test Geometry.get_element_lengths(geometry, 1) == (2.0,)
        @test Geometry.get_element_vertices(geometry, 1) == ((-1, 1),)
    end
end

@testset "2D" verbose=true begin
    @testset "2 patches, both Cartesian" verbose=true begin
        brk1 = (LinRange(0.5, 2.5, 5), LinRange(-0.75, 0.75, 3))
        brk2 = (LinRange(2.5, 5.0, 6), LinRange(-0.75, 0.75, 5))
        cg1 = Geometry.CartesianGeometry(brk1)
        cg2 = Geometry.CartesianGeometry(brk2)
        topology = Topology.MeshTopology(((1, 2, 3, 4), (2, 5, 6, 3)), Topology.QUAD)
        georef = Geometry.CartesianGeometry((brk1, brk2), topology)
        geometry = Geometry.UnstructuredGeometry((cg1, cg2), topology)

        # Geometry.jl
        @test Geometry.get_manifold_dim(geometry) == 2
        @test Geometry.get_image_dim(geometry) == 2
        @test Geometry.get_num_patches(geometry) == 2
        @test Geometry.get_topology(geometry) == geometry.topology
        @test Geometry.get_patch_id(geometry, 1) == 1
        @test Geometry.get_patch_and_local_element_id(geometry, 1) == (1, 1)
        @test Geometry.get_global_element_id(geometry, 1, 1) == 1
        @test Geometry.get_num_elements(geometry) == 28

        expected_num_elements = Dict(1 => 8, 2 => 20)
        @test @expected expected_num_elements k -> Geometry.get_num_elements(geometry, k)
        @test Geometry.get_num_elements_per_patch(geometry) == (8, 20)
        @test Geometry.get_element_measure(geometry, 1) == 0.375
        @test Geometry.get_element_lengths(geometry, 1) == (0.5, 0.75)
        @test Geometry.get_element_vertices(geometry, 1) == ((0.5, 1.0), (-0.75, 0.0))

        xi = Points.TensorProductPoints((LinRange(0.0, 1.0, 2), LinRange(0.0, 1.0, 2)))

        expected_evaluate = Dict{Int, Matrix{Float64}}()
        expected_jacobian = Dict{Int, Vector{SMatrix{2, 2, Float64, 4}}}()
        expected_hessian = Dict{Int, Vector{NTuple{2, SMatrix{2, 2, Float64, 4}}}}()
        for i in 1:Geometry.get_num_elements(georef)
            expected_evaluate[i] = Geometry.evaluate(georef, i, xi)
            expected_jacobian[i] = Geometry.jacobian(georef, i, xi)
            expected_hessian[i] = Geometry.hessian(georef, i, xi)
        end
        @test @expected expected_evaluate k -> Geometry.evaluate(geometry, k, xi)
        @test @expected expected_jacobian k -> Geometry.jacobian(geometry, k, xi)
        @test @expected expected_hessian k -> Geometry.hessian(geometry, k, xi)
    end
end

end
