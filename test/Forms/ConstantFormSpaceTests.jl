module ConstantFormSpaceTests

using Mantis
using Test
import LinearAlgebra

const L = 1.0
const num_elements_per_dim = 2
const num_points_per_dim = 3

############################################################################################
#                                         Cartesian geometry                               #
############################################################################################

function test_cartesian(manifold_dim::Int, num_quad_points_per_dim::Int)
    # Create a simple Cartesian geometry with random breakpoints
    breakpoints = ntuple(manifold_dim) do i
        el_sizes = 1.1 .^ (1:num_elements_per_dim)
        el_sizes ./= sum(el_sizes)
        el_sizes .*= L
        return cumsum([0.0; el_sizes])
    end
    geom = Geometry.CartesianGeometry(breakpoints)
    canonical_qrule = Quadrature.tensor_product_rule(
        ntuple(i->num_quad_points_per_dim, manifold_dim), Quadrature.gauss_legendre
    )
    dΩ = Mantis.Quadrature.StandardQuadrature(
        canonical_qrule, Geometry.get_num_elements(geom)
    )

    # Setup the form spaces, evaluate and check the results
    ξ = Points.CartesianPoints(ntuple(i->range(0.0, 1.0, num_points_per_dim), manifold_dim))
    for form_rank in 0:manifold_dim
        if form_rank ∉ Set([0, manifold_dim])
            @test_throws ArgumentError Forms.ConstantFormSpace(form_rank, geom, "c")
        else
            form_space = Forms.ConstantFormSpace(form_rank, geom, "a")

            # Check number of basis functions
            @test Forms.get_num_basis(form_space) == 1

            # Check number of basis functions for each element
            for element_id in 1:Forms.get_num_elements(form_space)
                @test Forms.get_num_basis(form_space, element_id) == 1
            end

            # Check max local dimension
            @test Forms.get_max_local_dim(form_space) == 1

            # Check estimated nnz per element
            @test Forms.get_estimated_nnz_per_elem(form_space) == 1

            # Check evaluate
            for element_id in 1:Forms.get_num_elements(form_space)
                eval, inds = Forms.evaluate(form_space, element_id, ξ)
                if form_rank == 0
                    @test eval == [reshape([1.0 for _ in 1:Points.get_num_points(ξ)], :, 1)]
                    @test inds == [[1]]
                elseif form_rank == manifold_dim
                    _, sqrt_g = Geometry.metric(geom, element_id, ξ)
                    @test isapprox(eval[1], reshape(sqrt_g, :, 1), atol=1e-12)
                    @test inds == [[1]]
                end
            end

            # Check derivative
            if form_rank == 0
                d_form_space = d(form_space)
                @test Forms.get_num_basis(d_form_space) == 1
                for element_id in 1:Forms.get_num_elements(form_space)
                    d_eval, d_inds = Forms.evaluate(d_form_space, element_id, ξ)
                    for i in 1:manifold_dim
                        @test d_eval[i] ==
                            reshape([0.0 for _ in 1:Points.get_num_points(ξ)], :, 1)
                    end
                    @test d_inds == [[1]]
                end
            end

            # Check integral
            if form_rank == 0
                integral = ∫(★(form_space), dΩ)
            else
                integral = ∫(form_space, dΩ)
            end
            integral_Eval = 0.0
            for element_id in 1:Forms.get_num_elements(form_space)
                integral_Eval += sum(Forms.evaluate(integral, element_id)[1])
            end
            @test isapprox(integral_Eval, L^manifold_dim, atol=1e-12)
        end
    end
end

@testset "Cartesian Geometry" begin
    for manifold_dim in 1:3
        for num_quad_points_per_dim in 1:5
            test_cartesian(manifold_dim, num_quad_points_per_dim)
        end
    end
end

############################################################################################
#                                         Mapped geometry                                  #
############################################################################################

const manifold_dim = 2
const r = 1.0
const Δr = 0.1

function mapping(x::AbstractVector)
    return [x[1] .* cos(x[2]*π/2), x[1] .* sin(x[2]*π/2), 1.0]
end
function dmapping(x::AbstractVector)
    return [
        cos(x[2]*π/2) -x[1]*sin(x[2]*π/2)*π/2;
        sin(x[2]*π/2) x[1]*cos(x[2]*π/2)*π/2;
        0.0 0.0
    ]
end
dimension = (2, 3)
curved_mapping = Geometry.Mapping(dimension, mapping, dmapping)

breakpoints = (LinRange(Δr, r, 4), LinRange(0.0, 1.0, 4))
geom = Geometry.CartesianGeometry(breakpoints)
mapped_geometry = Geometry.MappedGeometry(geom, curved_mapping)

surface_area = π * (r^2 - Δr^2)/4

function test_mapped(num_quad_points_per_dim::Int)
    canonical_qrule = Quadrature.tensor_product_rule(
        ntuple(i->num_quad_points_per_dim, manifold_dim), Quadrature.gauss_legendre
    )
    dΩ = Mantis.Quadrature.StandardQuadrature(
        canonical_qrule, Geometry.get_num_elements(mapped_geometry)
    )

    # Setup the form spaces, evaluate and check the results
    ξ = Points.CartesianPoints(ntuple(i->range(0.0, 1.0, num_points_per_dim), manifold_dim))
    for form_rank in 0:manifold_dim
        if form_rank ∉ Set([0, manifold_dim])
            @test_throws ArgumentError Forms.ConstantFormSpace(
                form_rank, mapped_geometry, "c"
            )
        else
            form_space = Forms.ConstantFormSpace(form_rank, mapped_geometry, "a")

            # Check number of basis functions
            @test Forms.get_num_basis(form_space) == 1

            # Check number of basis functions for each element
            for element_id in 1:Forms.get_num_elements(form_space)
                @test Forms.get_num_basis(form_space, element_id) == 1
            end

            # Check max local dimension
            @test Forms.get_max_local_dim(form_space) == 1

            # Check estimated nnz per element
            @test Forms.get_estimated_nnz_per_elem(form_space) == 1

            # Check evaluate
            for element_id in 1:Forms.get_num_elements(form_space)
                eval, inds = Forms.evaluate(form_space, element_id, ξ)
                if form_rank == 0
                    @test eval == [reshape([1.0 for _ in 1:Points.get_num_points(ξ)], :, 1)]
                    @test inds == [[1]]
                elseif form_rank == manifold_dim
                    _, sqrt_g = Geometry.metric(mapped_geometry, element_id, ξ)
                    @test isapprox(eval[1], reshape(sqrt_g, :, 1), atol=1e-12)
                    @test inds == [[1]]
                end
            end

            # Check derivative
            if form_rank == 0
                d_form_space = d(form_space)
                @test Forms.get_num_basis(d_form_space) == 1
                for element_id in 1:Forms.get_num_elements(form_space)
                    d_eval, d_inds = Forms.evaluate(d_form_space, element_id, ξ)
                    for i in 1:manifold_dim
                        @test d_eval[i] ==
                            reshape([0.0 for _ in 1:Points.get_num_points(ξ)], :, 1)
                    end
                    @test d_inds == [[1]]
                end
            end

            # Check integral
            if form_rank == 0
                integral = ∫(★(form_space), dΩ)
            else
                integral = ∫(form_space, dΩ)
            end
            integral_Eval = 0.0
            for element_id in 1:Forms.get_num_elements(form_space)
                integral_Eval += sum(Forms.evaluate(integral, element_id)[1])
            end
            @test isapprox(integral_Eval, surface_area, atol=1e-12)
        end
    end
end

@testset "Mapped Geometry" begin
    for num_quad_points_per_dim in 2:5
        test_mapped(num_quad_points_per_dim)
    end
end

end
