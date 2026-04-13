module IntegralTests

using Mantis
using Test

using LinearAlgebra
using SparseArrays

############################################################################################
##                                       Testing methods                                  ##
############################################################################################

function test_2d_evaluations(
    u⁰, u¹, u², dΩ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim}; atol
) where {manifold_dim}
    geometry = Forms.get_geometry(u⁰)
    f⁰ = Forms.AnalyticalFormField(0, scalar_valued_2d_func, geometry, "f⁰")
    f¹ = Forms.AnalyticalFormField(1, vector_valued_2d_func, geometry, "f¹")
    f² = Forms.AnalyticalFormField(2, scalar_valued_2d_func, geometry, "f²")

    ∫⁰ = ∫(u⁰ ∧ ★(u⁰), dΩ)
    ∫¹ = ∫(u¹ ∧ ★(u¹), dΩ)
    ∫² = ∫(u² ∧ ★(u²), dΩ)
    ∫f⁰ = ∫(f⁰ ∧ ★(f⁰), dΩ)
    ∫f¹ = ∫(f¹ ∧ ★(f¹), dΩ)
    ∫f² = ∫(f² ∧ ★(f²), dΩ)

    ∫⁰_eval = 0.0
    ∫f⁰_eval = 0.0
    ∫f¹_eval = 0.0
    ∫f²_eval = 0.0
    integrated_metric_1 = zeros(manifold_dim, manifold_dim)
    for element_id in 1:Forms.get_num_elements(u⁰)
        dΩₑ = Quadrature.get_element_quadrature_rule(dΩ, element_id)
        inv_g, _, det_g = Geometry.inv_metric(
            geometry, element_id, Quadrature.get_nodes(dΩₑ)
        )
        weights = Quadrature.get_weights(dΩₑ)
        integrated_metric_0 = dot(weights, det_g)
        curr_eval = sum(Forms.evaluate(∫⁰, element_id)[1])
        @test isapprox(integrated_metric_0, curr_eval, atol=1e-12)
        ∫⁰_eval += curr_eval
        element_lengths = [Geometry.get_element_lengths(geometry, element_id)...]
        integrated_metric_1 .= 0.0
        @inbounds for id in eachindex(weights)
            @. integrated_metric_1 += weights[id] * inv_g[id] * det_g[id]
        end

        reference_result = dot(element_lengths, integrated_metric_1 * element_lengths)
        @test isapprox(sum(Forms.evaluate(∫¹, element_id)[1]), reference_result, atol=1e-12)
        integrated_metric_2 = dot(weights, 1.0 ./ det_g)
        reference_result = integrated_metric_2 * prod(x -> x^2, element_lengths)
        @test isapprox(sum(Forms.evaluate(∫², element_id)[1]), reference_result, atol=1e-12)
        ∫f⁰_eval += Forms.evaluate(∫f⁰, element_id)[1][1]
        ∫f¹_eval += Forms.evaluate(∫f¹, element_id)[1][1]
        ∫f²_eval += Forms.evaluate(∫f², element_id)[1][1]
    end

    int_val = 16 * π^4
    @test isapprox(∫⁰_eval, 1.0, atol=atol)
    @test isapprox(∫f⁰_eval, int_val, atol=atol)
    @test isapprox(∫f¹_eval, 2 * int_val, atol=atol)
    @test isapprox(∫f²_eval, int_val, atol=atol)

    return nothing
end

function test_3d_evaluations(
    u⁰, u¹, dΩ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim}; atol
) where {manifold_dim}
    geometry = Forms.get_geometry(u⁰)
    f⁰ = Forms.AnalyticalFormField(0, scalar_valued_3d_func, geometry, "f⁰")
    f¹ = Forms.AnalyticalFormField(1, vector_valued_3d_func, geometry, "f¹")
    f³ = Forms.AnalyticalFormField(3, scalar_valued_3d_func, geometry, "f³")

    ∫⁰ = ∫(u⁰ ∧ ★(u⁰), dΩ)
    ∫¹ = ∫(u¹ ∧ ★(u¹), dΩ)
    ∫f⁰ = ∫(f⁰ ∧ ★(f⁰), dΩ)
    ∫f¹ = ∫(f¹ ∧ ★(f¹), dΩ)
    ∫f³ = ∫(f³ ∧ ★(f³), dΩ)

    ∫⁰_eval = 0.0
    ∫f⁰_eval = 0.0
    ∫f¹_eval = 0.0
    ∫f³_eval = 0.0
    integrated_metric_1 = zeros(manifold_dim, manifold_dim)
    for element_id in 1:Forms.get_num_elements(u⁰)
        dΩₑ = Quadrature.get_element_quadrature_rule(dΩ, element_id)
        inv_g, _, det_g = Geometry.inv_metric(
            geometry, element_id, Quadrature.get_nodes(dΩₑ)
        )
        weights = Quadrature.get_weights(dΩₑ)
        integrated_metric_0 = dot(weights, det_g)
        curr_eval = sum(Forms.evaluate(∫⁰, element_id)[1])
        @test isapprox(integrated_metric_0, curr_eval, atol=1e-12)
        ∫⁰_eval += curr_eval
        element_lengths = [Geometry.get_element_lengths(geometry, element_id)...]
        integrated_metric_1 .= 0.0
        @inbounds for id in eachindex(weights)
            @. integrated_metric_1 += weights[id] * inv_g[id] * det_g[id]
        end

        reference_result = dot(element_lengths, integrated_metric_1 * element_lengths)
        @test isapprox(sum(Forms.evaluate(∫¹, element_id)[1]), reference_result, atol=1e-12)
        ∫f⁰_eval += Forms.evaluate(∫f⁰, element_id)[1][1]
        ∫f¹_eval += Forms.evaluate(∫f¹, element_id)[1][1]
        ∫f³_eval += Forms.evaluate(∫f³, element_id)[1][1]
    end

    int_val = 8 * π^4
    @test isapprox(∫⁰_eval, 1.0, atol=atol)
    @test isapprox(∫f⁰_eval, int_val, atol=atol)
    @test isapprox(∫f¹_eval, 3 * int_val, atol=atol)
    @test isapprox(∫f³_eval, int_val, atol=atol)

    return nothing
end

############################################################################################
##                                       Variables setup                                  ##
############################################################################################
# Domains.
const starting_point_2d = (0.0, 0.0)
const box_size_2d = (1.0, 1.0)
const num_elements_2d = (3, 4)
const starting_point_3d = (0.0, 0.0, 0.0)
const box_size_3d = (1.0, 1.0, 1.0)
const num_elements_3d = (3, 4, 5)
const c = 0.2

# Polynomial degrees.
const degrees_2d = (2, 3)
const regularities_2d = (degrees_2d[1] - 1, degrees_2d[2] - 1)
const degrees_3d = (2, 3, 1)
const regularities_3d = (degrees_3d[1] - 1, degrees_3d[2] - 1, degrees_3d[3] - 1)

function scalar_valued_2d_func(x::Matrix{Float64})
    return [@. 8.0 * pi^2 * sinpi(2.0 * x[:, 1]) * sinpi(2.0 * x[:, 2])]
end

function vector_valued_2d_func(x::Matrix{Float64})
    return @. [
        8.0 * pi^2 * sinpi(2.0 * x[:, 1]) * sinpi(2.0 * x[:, 2]),
        8.0 * pi^2 * sinpi(2.0 * x[:, 1]) * sinpi(2.0 * x[:, 2]),
    ]
end

function scalar_valued_3d_func(x::Matrix{Float64})
    return [
        @. 8.0 * pi^2 * sinpi(2.0 * x[:, 1]) * sinpi(2.0 * x[:, 2]) * sinpi(2.0 * x[:, 3])
    ]
end

function vector_valued_3d_func(x::Matrix{Float64})
    return @. [
        8.0 * pi^2 * sinpi(2.0 * x[:, 1]) * sinpi(2.0 * x[:, 2]) * sinpi(2.0 * x[:, 3]),
        8.0 * pi^2 * sinpi(2.0 * x[:, 1]) * sinpi(2.0 * x[:, 2]) * sinpi(2.0 * x[:, 3]),
        8.0 * pi^2 * sinpi(2.0 * x[:, 1]) * sinpi(2.0 * x[:, 2]) * sinpi(2.0 * x[:, 3]),
    ]
end

############################################################################################
#                                            2D                                            #
############################################################################################
# Setup the complex
cart_complex_2d = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point_2d, box_size_2d, num_elements_2d, degrees_2d, regularities_2d
)
curv_mapping = Geometry.create_curvilinear_mapping(starting_point_2d, box_size_2d, c)
curv_complex_2d = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point_2d,
    box_size_2d,
    num_elements_2d,
    map(FunctionSpaces.Bernstein, degrees_2d),
    regularities_2d,
    curv_mapping,
)

# The canonical quadrature information.
canonical_qrule_2d = Quadrature.tensor_product_rule(
    3 .* (degrees_2d .+ 1), Quadrature.gauss_legendre
)
dΩ₂ = Quadrature.StandardQuadrature(
    canonical_qrule_2d, Geometry.get_num_elements(Forms.get_geometry(cart_complex_2d...))
)

# Test the different geometries.
@testset "2D" verbose = true begin
    @testset "TensorProductGeometry" begin
        cart_u⁰ = cart_complex_2d[1]
        cart_u¹ = cart_complex_2d[2]
        cart_u² = cart_complex_2d[3]
        test_2d_evaluations(cart_u⁰, cart_u¹, cart_u², dΩ₂; atol=1e-9)
    end

    @testset "MappedGeometry" begin
        curv_u⁰ = curv_complex_2d[1]
        curv_u¹ = curv_complex_2d[2]
        curv_u² = curv_complex_2d[3]
        # Higher tolerance due to the trigonometric mapping
        test_2d_evaluations(curv_u⁰, curv_u¹, curv_u², dΩ₂; atol=1e-4)
    end
end

############################################################################################
#                                            3D                                            #
############################################################################################
# Setup the complexes
cart_complex_3d = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point_3d, box_size_3d, num_elements_3d, degrees_3d, regularities_3d
)
curv_mapping = Geometry.create_curvilinear_mapping(starting_point_3d, box_size_3d, c)
curv_complex_3d = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point_3d,
    box_size_3d,
    num_elements_3d,
    map(FunctionSpaces.Bernstein, degrees_3d),
    regularities_3d,
    curv_mapping,
)

# The quadrature information.
canonical_qrule_3d = Quadrature.tensor_product_rule(
    3 .* (degrees_3d .+ 1), Quadrature.gauss_legendre
)
dΩ₃ = Quadrature.StandardQuadrature(
    canonical_qrule_3d, Geometry.get_num_elements(Forms.get_geometry(cart_complex_3d...))
)
Plot.export_geometry_to_vtk(Forms.get_geometry(curv_complex_3d[1]), "curv_test")

# Test the different geometries.
@testset "3D" verbose = true begin
    @testset "TensorProductGeometry" begin
        cart_u⁰ = cart_complex_3d[1]
        cart_u¹ = cart_complex_3d[2]
        test_3d_evaluations(cart_u⁰, cart_u¹, dΩ₃; atol=1e-9)
    end

    @testset "MappedGeometry" begin
        curv_u⁰ = curv_complex_3d[1]
        curv_u¹ = curv_complex_3d[2]
        test_3d_evaluations(
            curv_u⁰,
            curv_u¹,
            dΩ₃;
            atol=1e-4, # Higher tolerance due to the trigonometric mapping
        )
    end
end

end
