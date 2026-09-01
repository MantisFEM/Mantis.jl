module RefinementTests

using Mantis
using Mantis.Hierarchical
using Test

const ref_2(g) = Geometry.refinement_uniform(g, 2)

@testset "Uniform" verbose = true begin
    struct NewGeo{manifold_dim, image_dim, num_patches} <:
           Geometry.AbstractGeometry{manifold_dim, image_dim, num_patches} end

    @test_throws MethodError Geometry.refinement_uniform(NewGeo{1, 1, 1}, (1,))

    @testset "Breakpoints" verbose = true begin
        @test_throws ArgumentError Geometry.refinement_uniform(Int[], 2)
        @test Geometry.refinement_uniform([1, 4, 10], 3) ==
            [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
        @test Geometry.refinement_uniform(LinRange(0, 1, 5), 2) ==
            LinRange(0, 1, 9)
        @test Geometry.refinement_uniform([1], 2) isa Vector{Float64}
        @test Geometry.refinement_uniform([1], 2; T=Float32) isa Vector{Float32}
    end

    @testset "Cartesian" verbose = true begin
        cart_1d_1 = Geometry.create_cartesian_box((0.0,), (1.0,), (1,))
        cart_1d_2 = ref_2(cart_1d_1)
        @test Geometry.get_num_elements(cart_1d_2) == 2
        @test Geometry.get_element_vertices(cart_1d_2, 1) == ((0.0, 0.5),)
        @test Geometry.get_element_vertices(cart_1d_2, 2) == ((0.5, 1.0),)
        cart_1d_2 = Refinement(cart_1d_1, ref_2)()
        @test Geometry.get_num_elements(cart_1d_2) == 2
        @test Geometry.get_element_vertices(cart_1d_2, 1) == ((0.0, 0.5),)
        @test Geometry.get_element_vertices(cart_1d_2, 2) == ((0.5, 1.0),)
        cart_1d_ref = Refinement(cart_1d_1, g -> g |> ref_2 |> ref_2)
        cart_1d_2 = cart_1d_ref()
        @test Geometry.get_num_elements(cart_1d_2) == 4
        @test Geometry.get_element_vertices(cart_1d_2, 1) == ((0.0, 0.25),)
        @test Geometry.get_element_vertices(cart_1d_2, 2) == ((0.25, 0.5),)
        @test Geometry.get_element_vertices(cart_1d_2, 3) == ((0.5, 0.75),)
        @test Geometry.get_element_vertices(cart_1d_2, 4) == ((0.75, 1.0),)

        cart_2d_1 = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (1, 1))
        cart_2d_2 = Refinement(cart_2d_1, g -> g |> ref_2 |> ref_2)()
        @test Geometry.get_num_elements(cart_2d_2) == 16
        @test Geometry.get_element_vertices(cart_2d_2, 1) == ((0.0, 0.25), (0.0, 0.25))
    end
end

end
