module WedgeTests

using Mantis
using Test

import LinearAlgebra
using SparseArrays

############################################################################################
##                                       Testing methods                                  ##
############################################################################################
function test_inner_prod_equality(
    complex, dΩ::Q
) where {manifold_dim, Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim}}
    for rank in 0:manifold_dim
        # Generate the form space ϵᵏ, or form rank k, and matching hodge and form field.
        ϵ = complex[rank + 1]
        ★ϵ = ★(ϵ)
        α⁰ = Forms.AnalyticalFormField(0, zero_form_expression, Forms.get_geometry(ϵ), "α⁰")
        ∫ϵ = ∫(ϵ ∧ ★(ϵ), dΩ)
        integral_triple = ∫((ϵ ∧ ★(ϵ)) ∧ α⁰, dΩ)

        for element_id in 1:Quadrature.get_num_base_elements(dΩ)
            # Compare the inner product
            #   <ϵ, ϵ>
            # with
            #   ∫ ϵ ∧ ★ϵ

            # Compute the integral: ∫ ϵ ∧ ★ϵ
            integral_eval, _ = Forms.evaluate(∫ϵ, element_id)
            # Test if ϵ ∧ ★ϵ == ϵ ∧ ★ϵ ∧ α⁰ (since α⁰ = 1)
            integral_triple_eval, _ = Forms.evaluate(integral_triple, element_id)
            @test all(isapprox.(integral_eval, integral_triple_eval; atol=1e-12))

            # Test if the wedge product expression has the correct expression rank
            @test Forms.get_expression_rank(Forms.get_form(∫ϵ)) == 2
        end

        # Test ϵ¹ ∧ ⋆ϵ¹ ∧ ϵ¹ to check if form rank 3 expressions are not allowed
        if rank == 1
            wedge = ϵ ∧ ★ϵ
            @test_throws ArgumentError wedge ∧ ϵ
        end

        # Test ϵⁿ ∧ ϵⁿ (volume forms) throws error
        if rank == manifold_dim
            @test_throws ArgumentError ϵ ∧ ϵ
        end
    end

    return nothing
end

function test_combinations_2d(complex, q_rule)

    # Create the form spaces.
    ϵ⁰ = complex[1]
    ϵ¹ = complex[2]
    ϵ² = complex[3]

    # Create the form fields.
    ε⁰ = Forms.FormField(ϵ⁰)
    ε¹ = Forms.FormField(ϵ¹)
    ε² = Forms.FormField(ϵ²)

    # Test if different form-/expression-rank combinations don't throw errors
    zero_form_space_wedge = Forms.Wedge(ϵ⁰, ϵ⁰) # 0-form space with 0-form space
    zero_form_field_wedge = Forms.Wedge(ε⁰, ε⁰) # 0-form field with 0-form field
    zero_form_mixed_1_wedge = Forms.Wedge(ϵ⁰, ε⁰) # 0-form space with 0-form field
    zero_form_mixed_2_wedge = Forms.Wedge(ε⁰, ϵ⁰) # 0-form field with 0-form space
    zero_top_form_space_wedge = Forms.Wedge(ϵ⁰, ϵ²) # 0-form space with top-form space
    top_zero_form_space_wedge = Forms.Wedge(ϵ², ϵ⁰) # top-form space with 0-form space
    zero_top_form_mixed_1_wedge = Forms.Wedge(ϵ⁰, ε²) # 0-form space with top-form field
    zero_top_form_mixed_2_wedge = Forms.Wedge(ε⁰, ϵ²) # 0-form field with top-form space
    top_zero_form_mixed_1_wedge = Forms.Wedge(ϵ², ε⁰) # top-form space with 0-form field
    top_zero_form_mixed_2_wedge = Forms.Wedge(ε², ϵ⁰) # top-form field with 0-form space
    top_zero_form_field_wedge = Forms.Wedge(ε², ε⁰) # top-form field with 0-form field
    zero_top_form_field_wedge = Forms.Wedge(ε⁰, ε²) # 0-form field with top-form field
    one_form_space_wedge = Forms.Wedge(ϵ¹, ϵ¹) # 1-form space with 1-form space
    one_form_field_wedge = Forms.Wedge(ε¹, ε¹) # 1-form field with 1-form field
    one_form_mixed_1_wedge = Forms.Wedge(ϵ¹, ε¹) # 1-form space with 1-form field
    one_form_mixed_2_wedge = Forms.Wedge(ε¹, ϵ¹) # 1-form field with 1-form space
    one_zero_form_space_wedge = Forms.Wedge(ϵ¹, ϵ⁰) # 1-form space with 0-form space
    one_zero_form_field_wedge = Forms.Wedge(ε¹, ε⁰) # 1-form field with 0-form field
    one_zero_form_mixed_1_wedge = Forms.Wedge(ϵ¹, ε⁰) # 1-form space with 0-form field
    one_zero_form_mixed_2_wedge = Forms.Wedge(ε¹, ϵ⁰) # 1-form field with 0-form space
    zero_one_form_space_wedge = Forms.Wedge(ϵ⁰, ϵ¹) # 0-form space with 1-form space
    zero_one_form_field_wedge = Forms.Wedge(ε⁰, ε¹) # 0-form field with 1-form field
    zero_one_form_mixed_1_wedge = Forms.Wedge(ϵ⁰, ε¹) # 0-form space with 1-form field
    zero_one_form_mixed_2_wedge = Forms.Wedge(ε⁰, ϵ¹) # 0-form field with 1-form space

    quad_nodes = Quadrature.get_nodes(Quadrature.get_element_quadrature_rule(q_rule, 1))

    # We changed the tests to use try/catch because @test_nowarn was extremely slow, as it
    # triggers a recompilation for every call.
    test_const = 0
    try
        Forms.evaluate(zero_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(zero_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(zero_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(zero_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(zero_top_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(top_zero_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(zero_top_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(zero_top_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(top_zero_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(top_zero_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(top_zero_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(zero_top_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(one_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(one_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(one_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(one_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(one_zero_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(one_zero_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(one_zero_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(one_zero_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(zero_one_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(zero_one_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(zero_one_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(zero_one_form_mixed_2_wedge, 1, quad_nodes)
    catch e
        test_const = 1
        println(e)
    end

    @test test_const == 0

    return nothing
end

function test_combinations_3d(complex, q_rule)

    # Create form spaces
    ϵ⁰ = complex[1]
    ϵ¹ = complex[2]
    ϵ² = complex[3]
    ϵ³ = complex[4]

    # Create the form fields
    ε⁰ = Forms.FormField(ϵ⁰)
    ε¹ = Forms.FormField(ϵ¹)
    ε² = Forms.FormField(ϵ²)
    ε³ = Forms.FormField(ϵ³)

    # Test if different form-/expression-rank combinations don't throw errors
    zero_form_space_wedge = Forms.Wedge(ϵ⁰, ϵ⁰) # 0-form space with 0-form space
    zero_form_field_wedge = Forms.Wedge(ε⁰, ε⁰) # 0-form field with 0-form field
    zero_form_mixed_1_wedge = Forms.Wedge(ϵ⁰, ε⁰) # 0-form space with 0-form field
    zero_form_mixed_2_wedge = Forms.Wedge(ε⁰, ϵ⁰) # 0-form field with 0-form space
    zero_one_form_space_wedge = Forms.Wedge(ϵ⁰, ϵ¹) # 0-form space with 1-form space
    zero_one_form_field_wedge = Forms.Wedge(ε⁰, ε¹) # 0-form field with 1-form field
    zero_one_form_mixed_1_wedge = Forms.Wedge(ϵ⁰, ε¹) # 0-form space with 1-form field
    zero_one_form_mixed_2_wedge = Forms.Wedge(ε⁰, ϵ¹) # 0-form field with 1-form space
    one_form_space_wedge = Forms.Wedge(ϵ¹, ϵ¹) # 1-form space with 1-form space
    one_form_field_wedge = Forms.Wedge(ε¹, ε¹) # 1-form field with 1-form field
    one_form_mixed_1_wedge = Forms.Wedge(ϵ¹, ε¹) # 1-form space with 1-form field
    one_form_mixed_2_wedge = Forms.Wedge(ε¹, ϵ¹) # 1-form field with 1-form space
    one_zero_form_space_wedge = Forms.Wedge(ϵ¹, ϵ⁰) # 1-form space with 0-form space
    one_zero_form_field_wedge = Forms.Wedge(ε¹, ε⁰) # 1-form field with 0-form field
    one_zero_form_mixed_1_wedge = Forms.Wedge(ϵ¹, ε⁰) # 1-form space with 0-form field
    one_zero_form_mixed_2_wedge = Forms.Wedge(ε¹, ϵ⁰) # 1-form field with 0-form space
    zero_two_form_space_wedge = Forms.Wedge(ϵ⁰, ϵ²) # 0-form space with 2-form space
    zero_two_form_field_wedge = Forms.Wedge(ε⁰, ε²) # 0-form field with 2-form field
    zero_two_form_mixed_1_wedge = Forms.Wedge(ϵ⁰, ε²) # 0-form space with 2-form field
    zero_two_form_mixed_2_wedge = Forms.Wedge(ε⁰, ϵ²) # 0-form field with 2-form space
    two_zero_form_space_wedge = Forms.Wedge(ϵ², ϵ⁰) # 2-form space with 0-form space
    two_zero_form_field_wedge = Forms.Wedge(ε², ε⁰) # 2-form field with 0-form field
    two_zero_form_mixed_1_wedge = Forms.Wedge(ϵ², ε⁰) # 2-form space with 0-form field
    two_zero_form_mixed_2_wedge = Forms.Wedge(ε², ϵ⁰) # 2-form field with 0-form space
    one_two_form_space_wedge = Forms.Wedge(ϵ¹, ϵ²) # 1-form space with 2-form space
    one_two_form_field_wedge = Forms.Wedge(ε¹, ε²) # 1-form field with 2-form field
    one_two_form_mixed_1_wedge = Forms.Wedge(ϵ¹, ε²) # 1-form space with 2-form field
    one_two_form_mixed_2_wedge = Forms.Wedge(ε¹, ϵ²) # 1-form field with 2-form space
    two_one_form_space_wedge = Forms.Wedge(ϵ², ϵ¹) # 2-form space with 1-form space
    two_one_form_field_wedge = Forms.Wedge(ε², ε¹) # 2-form field with 1-form field
    two_one_form_mixed_1_wedge = Forms.Wedge(ϵ², ε¹) # 2-form space with 1-form field
    two_one_form_mixed_2_wedge = Forms.Wedge(ε², ϵ¹) # 2-form field with 1-form space
    zero_top_form_space_wedge = Forms.Wedge(ϵ⁰, ϵ³) # 0-form space with top-form space
    zero_top_form_field_wedge = Forms.Wedge(ε⁰, ε³) # 0-form field with top-form field
    zero_top_form_mixed_1_wedge = Forms.Wedge(ϵ⁰, ε³) # 0-form space with top-form field
    zero_top_form_mixed_2_wedge = Forms.Wedge(ε⁰, ϵ³) # 0-form field with top-form space
    top_zero_form_space_wedge = Forms.Wedge(ϵ³, ϵ⁰) # top-form space with 0-form space
    top_zero_form_field_wedge = Forms.Wedge(ε³, ε⁰) # top-form field with 0-form field
    top_zero_form_mixed_1_wedge = Forms.Wedge(ϵ³, ε⁰) # top-form space with 0-form field
    top_zero_form_mixed_2_wedge = Forms.Wedge(ε³, ϵ⁰) # top-form field with 0-form space

    quad_nodes = Quadrature.get_nodes(Quadrature.get_element_quadrature_rule(q_rule, 1))

    # We changed the tests to use try/catch because @test_nowarn was extremely slow, as it
    # triggers a recompilation for every call.
    test_const = 0
    try
        Forms.evaluate(zero_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(zero_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(zero_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(zero_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(zero_one_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(zero_one_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(zero_one_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(zero_one_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(one_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(one_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(one_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(one_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(one_zero_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(one_zero_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(one_zero_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(one_zero_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(zero_two_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(zero_two_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(zero_two_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(zero_two_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(two_zero_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(two_zero_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(two_zero_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(two_zero_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(one_two_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(one_two_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(one_two_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(one_two_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(two_one_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(two_one_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(two_one_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(two_one_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(zero_top_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(zero_top_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(zero_top_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(zero_top_form_mixed_2_wedge, 1, quad_nodes)
        Forms.evaluate(top_zero_form_space_wedge, 1, quad_nodes)
        Forms.evaluate(top_zero_form_field_wedge, 1, quad_nodes)
        Forms.evaluate(top_zero_form_mixed_1_wedge, 1, quad_nodes)
        Forms.evaluate(top_zero_form_mixed_2_wedge, 1, quad_nodes)
    catch e
        test_const = 1
        println(e)
    end

    @test test_const == 0

    return nothing
end

# Oriented basis of the components of a `form_rank`-form, in the order used by Mantis.
function get_component_basis(manifold_dim::Int, form_rank::Int)
    if form_rank == 0
        return [Int[]]
    elseif form_rank == manifold_dim
        return [collect(1:manifold_dim)]
    elseif form_rank == 1
        return [[i] for i in 1:manifold_dim]
    end

    # 2-forms in 3D: dξ²∧dξ³, dξ³∧dξ¹, dξ¹∧dξ².
    return [[2, 3], [3, 1], [1, 2]]
end

# Sign of the permutation that sorts `indices` (which must be distinct).
function get_permutation_sign(indices::Vector{Int})
    num_inversions = 0
    for i in eachindex(indices), j in eachindex(indices)
        if i < j && indices[i] > indices[j]
            num_inversions += 1
        end
    end

    return iseven(num_inversions) ? 1 : -1
end

# Wedge of the evaluations of a `form_rank_1`-form and a `form_rank_2`-form, computed
# independently of `Forms.Wedge` from dξᴵ ∧ dξᴶ = sign(I, J) dξᴷ, with K the sorted I ∪ J.
function compute_reference_wedge(
    α_eval, β_eval, manifold_dim::Int, form_rank_1::Int, form_rank_2::Int
)
    basis_1 = get_component_basis(manifold_dim, form_rank_1)
    basis_2 = get_component_basis(manifold_dim, form_rank_2)
    basis_wedge = get_component_basis(manifold_dim, form_rank_1 + form_rank_2)
    wedge_eval = [zeros(length(α_eval[1])) for _ in basis_wedge]
    for (i, I) in enumerate(basis_1), (j, J) in enumerate(basis_2)
        if !isempty(intersect(I, J))
            continue
        end

        K = vcat(I, J)
        k = findfirst(B -> sort(B) == sort(K), basis_wedge)
        sign = get_permutation_sign(K) * get_permutation_sign(basis_wedge[k])
        wedge_eval[k] .+= sign .* α_eval[i] .* β_eval[j]
    end

    return wedge_eval
end

# Contracts every basis dimension `e` of a component evaluation with the coefficients of the
# basis functions given by `indices[e]`.
function contract_with_coefficients(values, indices, coefficients)
    expression_rank = length(coefficients)
    if expression_rank == 0
        return vec(values)
    end

    result = zeros(size(values, 1))
    for basis_id in CartesianIndices(size(values)[2:end])
        weight = prod(coefficients[e][indices[e][basis_id[e]]] for e in 1:expression_rank)
        result .+= weight .* values[:, Tuple(basis_id)...]
    end

    return result
end

# Checks the values of the wedge of all combinations of form ranks against a reference
# computed from form fields. Form spaces are checked by contracting their evaluation with
# the coefficients of the fields, which also checks the indices of every component.
function test_wedge_values(
    complex, dΩ::Q
) where {manifold_dim, Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim}}
    quad_nodes = Quadrature.get_nodes(Quadrature.get_element_quadrature_rule(dΩ, 1))
    num_elements = Forms.get_num_elements(complex[1])
    for form_rank_1 in 0:manifold_dim, form_rank_2 in 0:(manifold_dim - form_rank_1)
        ϵ¹ = complex[form_rank_1 + 1]
        ϵ² = complex[form_rank_2 + 1]
        coefficients_1 = sin.(1:Forms.get_num_basis(ϵ¹))
        coefficients_2 = cos.(1:Forms.get_num_basis(ϵ²))
        α = Forms.FormField(ϵ¹, coefficients_1, "α")
        β = Forms.FormField(ϵ², coefficients_2, "β")
        wedges = (
            (Forms.Wedge(α, β), ()),
            (Forms.Wedge(ϵ¹, β), (coefficients_1,)),
            (Forms.Wedge(α, ϵ²), (coefficients_2,)),
            (Forms.Wedge(ϵ¹, ϵ²), (coefficients_1, coefficients_2)),
        )
        for element_id in (1, num_elements)
            α_eval, _ = Forms.evaluate(α, element_id, quad_nodes)
            β_eval, _ = Forms.evaluate(β, element_id, quad_nodes)
            reference = compute_reference_wedge(
                α_eval, β_eval, manifold_dim, form_rank_1, form_rank_2
            )
            for (wedge, coefficients) in wedges
                wedge_eval, wedge_indices = Forms.evaluate(wedge, element_id, quad_nodes)
                @test length(wedge_eval) == length(reference)
                for component_id in eachindex(reference)
                    @test isapprox(
                        contract_with_coefficients(
                            wedge_eval[component_id],
                            wedge_indices[component_id],
                            coefficients,
                        ),
                        reference[component_id];
                        atol=1e-10,
                    )
                end
            end
        end
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
const degrees_2d = (2, 3)
const regularities_2d = (degrees_2d[1] - 1, degrees_2d[2] - 1)
const c = 0.2

# Expression for analytical form fields
function zero_form_expression(x::Matrix{Float64})
    return [ones(Float64, size(x))]
end

############################################################################################
#                                            2D                                            #
############################################################################################
# Setup the complex
cart_complex_2d = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point_2d, box_size_2d, num_elements_2d, degrees_2d, regularities_2d
)
curv_complex_2d = Forms.create_curvilinear_tensor_product_bspline_de_rham_complex(
    starting_point_2d, box_size_2d, num_elements_2d, degrees_2d, regularities_2d; c=c
)

# The canonical quadrature information.
canonical_qrule_2d = Quadrature.tensor_product_rule(
    degrees_2d .+ 2, Quadrature.gauss_legendre
)
dΩ₂ = Quadrature.StandardQuadrature(
    canonical_qrule_2d, Geometry.get_num_elements(Forms.get_geometry(cart_complex_2d...))
)

# Test the different geometries.
#
@testset "2D" verbose = true begin
    @testset "CartesianGeometry" begin
        test_inner_prod_equality(cart_complex_2d, dΩ₂)
        test_combinations_2d(cart_complex_2d, dΩ₂)
        test_wedge_values(cart_complex_2d, dΩ₂)
    end

    @testset "MappedGeometry" begin
        test_inner_prod_equality(curv_complex_2d, dΩ₂)
        test_combinations_2d(curv_complex_2d, dΩ₂)
        test_wedge_values(curv_complex_2d, dΩ₂)
    end
end

############################################################################################
#                                            3D                                            #
############################################################################################
#
# Polynomial degrees.
const starting_point_3d = (0.0, 0.0, 0.0)
const box_size_3d = (1.0, 1.0, 1.0)
const num_elements_3d = (3, 4, 5)
const degrees_3d = (2, 3, 1)
const regularities_3d = (degrees_3d[1] - 1, degrees_3d[2] - 1, degrees_3d[3] - 1)

# Setup the complex
cart_complex_3d = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point_3d, box_size_3d, num_elements_3d, degrees_3d, regularities_3d
)

# Crazy mesh geometry.
breakpoints3 = LinRange(
    starting_point_3d[3], starting_point_3d[3] + box_size_3d[3], num_elements_3d[3] + 1
)
line_geo_3 = Geometry.CartesianGeometry((breakpoints3,))
curv_geom_3d = Geometry.TensorProductGeometry((
    Forms.get_geometry(curv_complex_2d[1]), line_geo_3
))
curv_geom_3d_parametric = Geometry.TensorProductGeometry((
    Forms.get_geometry(cart_complex_2d[1]), line_geo_3
))
bsp_3 = FunctionSpaces.get_factor_spaces(Forms.get_fe_space(cart_complex_3d[1]))[3]
tp_space_curv_3d = FunctionSpaces.TensorProductSpace((
    Forms.get_fe_space(curv_complex_2d[1]), bsp_3
))
tp_space_curv_3d_ds = FunctionSpaces.DirectSumSpace((
    tp_space_curv_3d, tp_space_curv_3d, tp_space_curv_3d
))
curv_complex_3d = (
    Forms.FormSpace(0, tp_space_curv_3d, "w⁰"),
    Forms.FormSpace(1, tp_space_curv_3d_ds, "w¹"),
    Forms.FormSpace(2, tp_space_curv_3d_ds, "w²"),
    Forms.FormSpace(3, tp_space_curv_3d, "w³"),
)

# The quadrature information.
canonical_qrule_3d = Quadrature.tensor_product_rule(
    degrees_3d .+ 2, Quadrature.gauss_legendre
)
dΩ₃ = Quadrature.StandardQuadrature(
    canonical_qrule_3d, Geometry.get_num_elements(Forms.get_geometry(cart_complex_3d...))
)

# Test the different geometries.
@testset "3D" verbose = true begin
    @testset "CartesianGeometry" begin
        test_inner_prod_equality(cart_complex_3d, dΩ₃)
        test_combinations_3d(cart_complex_3d, dΩ₃)
        test_wedge_values(cart_complex_3d, dΩ₃)
    end

    @testset "MappedGeometry" begin
        test_inner_prod_equality(curv_complex_3d, dΩ₃)
        test_combinations_3d(curv_complex_3d, dΩ₃)
        test_wedge_values(curv_complex_3d, dΩ₃)
    end
end

end
