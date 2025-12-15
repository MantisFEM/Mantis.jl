module IntegralTests

using Mantis
using Test

using LinearAlgebra
using SparseArrays

############################################################################################
##                                       Testing methods                                  ##
############################################################################################

function test_evaluations(
    complex,
    dΩ::Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
    one_valued_func::Function,
    two_valued_func::Function,
) where {manifold_dim}
    geometry = Forms.get_geometry(complex...)
    u⁰ = complex[1]
    v¹ = complex[2]
    σ2 = complex[3]
    f⁰ = Forms.AnalyticalFormField(0, one_valued_func, geometry, "f⁰")
    f¹ = Forms.AnalyticalFormField(1, two_valued_func, geometry, "f¹")
    if manifold_dim == 2
        f² = Forms.AnalyticalFormField(2, one_valued_func, geometry, "f²")
    else
        f² = Forms.AnalyticalFormField(2, two_valued_func, geometry, "f²")
    end

    if manifold_dim == 3
        f³ = Forms.AnalyticalFormField(3, one_valued_func, geometry, "f³")
    end

    ∫⁰ = ∫(u⁰ ∧ ★(u⁰), dΩ)
    ∫¹ = ∫(v¹ ∧ ★(v¹), dΩ)
    ∫² = ∫(σ2 ∧ ★(σ2), dΩ)
    ∫d⁰ = ∫(d(u⁰) ∧ ★(d(u⁰)), dΩ)
    ∫d¹ = ∫(d(v¹) ∧ ★(d(v¹)), dΩ)
    ∫★⁰ = ∫(★(u⁰) ∧ ★(★(u⁰)), dΩ)
    ∫★¹ = ∫(★(v¹) ∧ ★(★(v¹)), dΩ)
    ∫★² = ∫(★(σ2) ∧ ★(★(σ2)), dΩ)
    ∫f⁰ = ∫(f⁰ ∧ ★(f⁰), dΩ)
    ∫f¹ = ∫(f¹ ∧ ★(f¹), dΩ)
    ∫f² = ∫(f² ∧ ★(f²), dΩ)
    if manifold_dim == 3
        ∫f³ = ∫(f³ ∧ ★(f³), dΩ)
    end

    ∫⁰_eval = 0.0
    ∫f⁰_eval = 0.0
    ∫f¹_eval = 0.0
    ∫f²_eval = 0.0
    ∫f³_eval = 0.0
    for element_id in 1:Forms.get_num_elements(u⁰)
        dΩₑ = Quadrature.get_element_quadrature_rule(dΩ, element_id)
        inv_g, g, det_g = Geometry.inv_metric(
            geometry, element_id, Quadrature.get_nodes(dΩₑ)
        )
        integrated_metric_0 = sum(Quadrature.get_weights(dΩₑ) .* det_g)
        curr_eval = sum(Forms.evaluate(∫⁰, element_id)[1])
        @test isapprox(integrated_metric_0, curr_eval, atol=1e-12)
        ∫⁰_eval += curr_eval
        element_lengths = [Geometry.get_element_lengths(geometry, element_id)...]
        integrated_metric_1 = zeros((manifold_dim, manifold_dim))
        for id in eachindex(Quadrature.get_weights(dΩₑ))
            integrated_metric_1 .+=
                Quadrature.get_weights(dΩₑ)[id] .* (inv_g[id] .* det_g[id])
        end
        reference_result = dot(element_lengths, integrated_metric_1 * element_lengths)
        @test isapprox(sum(Forms.evaluate(∫¹, element_id)[1]), reference_result, atol=1e-12)
        reference_result = 0.0
        @test isapprox(
            sum(Forms.evaluate(∫d⁰, element_id)[1]), reference_result, atol=1e-12
        )
        if manifold_dim == 2
            integrated_metric_2 = sum(Quadrature.get_weights(dΩₑ) .* (1.0 ./ det_g))
            reference_result = integrated_metric_2 * prod(element_lengths .^ 2)
            @test isapprox(
                sum(Forms.evaluate(∫², element_id)[1]), reference_result, atol=1e-12
            )
            reference_result = 0.0
            @test isapprox(
                sum(Forms.evaluate(∫d¹, element_id)[1]), reference_result, atol=1e-12
            )
        end

        @test all(
            isapprox.(
                Forms.evaluate(∫★⁰, element_id)[1],
                Forms.evaluate(∫⁰, element_id)[1],
                atol=1e-12,
            ),
        )
        @test all(
            isapprox.(
                Forms.evaluate(∫★¹, element_id)[1],
                Forms.evaluate(∫¹, element_id)[1],
                atol=1e-12,
            ),
        )
        @test all(
            isapprox.(
                Forms.evaluate(∫★², element_id)[1],
                Forms.evaluate(∫², element_id)[1],
                atol=1e-12,
            ),
        )
        ∫f⁰_eval += Forms.evaluate(∫f⁰, element_id)[1][1]
        ∫f¹_eval += Forms.evaluate(∫f¹, element_id)[1][1]
        if manifold_dim == 2
            ∫f²_eval += Forms.evaluate(∫f², element_id)[1][1]
        else
            ∫f³_eval += Forms.evaluate(∫f³, element_id)[1][1]
        end
    end

    @test isapprox(∫⁰_eval, 1.0, atol=1e-9)
    @test isapprox(∫f⁰_eval, 16 * π^4, atol=1e-9)
    @test isapprox(∫f¹_eval, 32 * π^4, atol=1e-9)
    if manifold_dim == 2
        @test isapprox(∫f²_eval, 16 * π^4, atol=1e-9)
    else
        @test isapprox(∫f³_eval, 16 * π^4, atol=1e-9)
    end

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
const crazy_c = 0.2

# Polynomial degrees.
const degrees_2d = (2, 3)
const regularities_2d = (degrees_2d[1] - 1, degrees_2d[2] - 1)
const degrees_3d = (2, 3, 1)
const regularities_3d = (degrees_3d[1] - 1, degrees_3d[2] - 1, degrees_3d[3] - 1)

# Expression for analytical form fields
function zero_form_expression(x::Matrix{Float64})
    return [ones(Float64, size(x))]
end

function analytical_1valued_form_func(x::Matrix{Float64})
    return [@. 8.0 * pi^2 * sinpi(2.0 * x[:, 1]) * sinpi(2.0 * x[:, 2])]
end

function analytical_2valued_form_func(x::Matrix{Float64})
    return @. [
        8.0 * pi^2 * sinpi(2.0 * x[:, 1]) * sinpi(2.0 * x[:, 2]),
        8.0 * pi^2 * sinpi(2.0 * x[:, 1]) * sinpi(2.0 * x[:, 2]),
    ]
end

############################################################################################
##                                       2D TESTS                                         ##
############################################################################################
# Setup the complex
cart_complex_2d = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point_2d, box_size_2d, num_elements_2d, degrees_2d, regularities_2d
)
curv_complex_2d = Forms.create_curvilinear_tensor_product_bspline_de_rham_complex(
    starting_point_2d,
    box_size_2d,
    num_elements_2d,
    degrees_2d,
    regularities_2d;
    crazy_c=crazy_c,
)

# The canonical quadrature information.
canonical_qrule_2d = Quadrature.tensor_product_rule(
    degrees_2d .+ 12, Quadrature.gauss_legendre
)
dΩ₂ = Quadrature.StandardQuadrature(
    canonical_qrule_2d, Geometry.get_num_elements(Forms.get_geometry(cart_complex_2d...))
)

# Test the different geometries.
@testset "2D" verbose = true begin
    @testset "TensorProductGeometry" begin
        test_evaluations(
            cart_complex_2d, dΩ₂, analytical_1valued_form_func, analytical_2valued_form_func
        )
    end

    @testset "MappedGeometry" begin
        test_evaluations(
            curv_complex_2d, dΩ₂, analytical_1valued_form_func, analytical_2valued_form_func
        )
    end
end

############################################################################################
##                                       3D TESTS                                         ##
############################################################################################
# Setup the complexes
cart_complex_3d = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point_3d, box_size_3d, num_elements_3d, degrees_3d, regularities_3d
)
curv_mapping = Geometry.create_curvilinear_mapping(starting_point_3d, box_size_3d, crazy_c)
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
    2 .* (degrees_3d .+ 1), Quadrature.gauss_legendre
)
dΩ₃ = Quadrature.StandardQuadrature(
    canonical_qrule_3d, Geometry.get_num_elements(Forms.get_geometry(cart_complex_3d...))
)
Plot.export_geometry_to_vtk(Forms.get_geometry(curv_complex_3d[1]), "curv_test")

# Test the different geometries.
@testset "3D" verbose = true begin
    @testset "TensorProductGeometry" begin
        test_evaluations(
            cart_complex_3d, dΩ₃, analytical_1valued_form_func, analytical_2valued_form_func
        )
    end

    # @testset "MappedGeometry" begin
    #     test_evaluations(
    #         curv_complex_3d, dΩ₃, analytical_1valued_form_func, analytical_2valued_form_func
    #     )
    # end
end

end
