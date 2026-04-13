module GeometryInferenceTests

import Pkg

using Mantis
using Memoization
using Test
using JET

# Test spaces

# Univariate ---
# BSplines
# Reduction test, single-patch, single element, 1D, Cartesian, degree 1 BSplines.
geometry1 = Geometry.CartesianGeometry(([-1, 1],))
B1 = FunctionSpaces.BSplineSpace(
    geometry1, geometry1, FunctionSpaces.Bernstein(1), [-1, -1], 1, 1
)
# Multi-element, Cartesian, degree 6 maximally smooth BSplines.
geometry1multi = Geometry.CartesianGeometry((LinRange(-0.34, 1.56, 26),))
B1multi = FunctionSpaces.BSplineSpace(geometry1multi, 6, 5)
# Multi-element, Cartesian, different Section Spaces.
B1LL = FunctionSpaces.BSplineSpace(
    geometry1, geometry1multi, FunctionSpaces.LobattoLegendre(1), fill(-1, 26)
)
B1GL = FunctionSpaces.BSplineSpace(
    geometry1, geometry1multi, FunctionSpaces.GaussLegendre(1), fill(-1, 26)
)
B1ELL = FunctionSpaces.BSplineSpace(
    geometry1, geometry1multi, FunctionSpaces.EdgeLobattoLegendre(1), fill(-1, 26)
)

# Rational
R1 = FunctionSpaces.RationalFESpace(B1, [0.2, 0.8])
R1m = FunctionSpaces.RationalFESpace(
    B1multi, rand(FunctionSpaces.get_num_basis(B1multi))
)

# Multi-variate and multi-component ---
# TensorProduct
# Reduction test, single-space, single-patch, single element, 1D
TP_B1 = FunctionSpaces.TensorProductSpace((B1,))
# Single-space, single-patch, multi element, 1D
TP_B1m = FunctionSpaces.TensorProductSpace((B1multi,))
# Two-space (one unique, Bspline), single-patch, multi element, 2D
TP_B1mB1m = FunctionSpaces.TensorProductSpace((B1multi, B1multi))
# Three-space (one unique, Bspline), single-patch, multi element, 3D
TP_B1mB1mB1m = FunctionSpaces.TensorProductSpace((B1multi, B1multi, B1multi))
# Two-space (one unique, Rational), single-patch, multi element, 2D
TP_R1mR1m = FunctionSpaces.TensorProductSpace((R1m, R1m))
# Three-space (one unique, Rational), single-patch, multi element, 3D
TP_R1mR1mR1m = FunctionSpaces.TensorProductSpace((R1m, R1m, R1m))
# Two-space (Bspline, Rational), single-patch, multi element, 2D
TP_B1mR1m = FunctionSpaces.TensorProductSpace((B1multi, R1m))

# DirectSumSpace
# Reduction test, single-component, single-patch, single element, 1D.
DS1 = FunctionSpaces.DirectSumSpace((B1,))
# 2-component, same space per component, 1D, multi-element, single-patch
DS_B1mB1m = FunctionSpaces.DirectSumSpace((B1multi, B1multi))
# 2-component, same space per component, 2D, multi-element, single-patch
DS_TP1mTP1m = FunctionSpaces.DirectSumSpace((TP_B1mB1m, TP_B1mB1m))
# 3-component, two distinct spaces, 2D, multi-element, single-patch
DS_TP1mTPR1mTP1m = FunctionSpaces.DirectSumSpace((TP_B1mB1m, TP_R1mR1m, TP_B1mB1m))
# 3-component, three distinct spaces, 2D, multi-element, single-patch
DS_TPB1mTPR1mTPBR1m = FunctionSpaces.DirectSumSpace((
    TP_B1mB1m, TP_R1mR1m, TP_B1mR1m
))

const spaces = (
    B1,
    B1multi,
    B1LL,
    B1GL,
    B1ELL,
    R1,
    TP_B1,
    TP_B1m,
    TP_B1mB1m,
    TP_B1mB1mB1m,
    TP_R1mR1m,
    TP_R1mR1mR1m,
    TP_B1mR1m,
    DS1,
    DS_B1mB1m,
    DS_TP1mTP1m,
    DS_TP1mTPR1mTP1m,
    DS_TPB1mTPR1mTPBR1m,
)

# Note that JET only uses the types of the inputs, so which numbers we pick
# here is irrelevant.
const xi_1D = Points.CartesianPoints(([0.0, 1.0],))
const xi_2D = Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0]))
const xi_3D = Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))

const element_id = 1
const component_id = 1
const nderivatives = 1
const basis_id = 4

foreach(spaces) do space

    # Methods on the type.
    @test_opt FunctionSpaces.get_manifold_dim(space)
    @test_opt FunctionSpaces.get_num_components(space)
    @test_opt FunctionSpaces.get_num_patches(space)

    # Methods which have a general fallback (can be an error fallback).
    @test_opt FunctionSpaces.get_component_spaces(space)
    @test_opt FunctionSpaces.get_extraction_operator(space)
    @test_opt FunctionSpaces.get_extraction(space, element_id, component_id)
    @test_opt FunctionSpaces.get_extraction_coefficients(
        space, element_id, component_id
    )
    @test_opt FunctionSpaces.get_basis_indices(space, element_id)
    @test_opt FunctionSpaces.get_basis_permutation(space, element_id, component_id)
    @test_opt FunctionSpaces.get_num_basis(space)
    @test_opt FunctionSpaces.get_num_basis(space, element_id)
    @test_opt FunctionSpaces.get_dof_partition(space)
    @test_opt FunctionSpaces.get_max_local_dim(space)
    @test_opt FunctionSpaces.get_geometry(space)
    @test_opt FunctionSpaces.get_parametric_geometry(space)

    # Methods specific to some spaces.
    if typeof(space) <: FunctionSpaces.BSplineSpace
        @test_opt FunctionSpaces.get_polynomials(space)
        @test_opt FunctionSpaces.get_polynomial_degree(space)
        @test_opt FunctionSpaces.get_multiplicity_vector(space)
        @test_opt FunctionSpaces.get_support(space, element_id)

    elseif typeof(space) <: FunctionSpaces.RationalFESpace
        @test_opt FunctionSpaces.get_polynomial_degree(space, element_id)

    elseif typeof(space) <: FunctionSpaces.TensorProductSpace
        @test_opt FunctionSpaces.get_cart_num_basis(space)
        @test_opt FunctionSpaces.get_lin_num_basis(space)
        @test_opt FunctionSpaces.get_constituent_spaces(space)
        @test_opt FunctionSpaces.get_cart_num_elements(space)
        @test_opt FunctionSpaces.get_lin_num_elements(space)
        @test_opt FunctionSpaces.get_num_spaces(space)
        @test_opt FunctionSpaces.get_support(space, basis_id)

        @test_opt FunctionSpaces.get_constituent_element_id(space, element_id)
        @test_opt FunctionSpaces.get_constituent_basis_id(space, basis_id)
        @test_opt FunctionSpaces.get_constituent_num_basis(space)
        @test_opt FunctionSpaces.get_constituent_num_basis(space, element_id)
        @test_opt FunctionSpaces.get_constituent_manifold_dim(space)
        @test_opt FunctionSpaces.get_constituent_basis_indices(space, element_id)
        @test_opt FunctionSpaces.get_constituent_support(space, element_id)
        @test_opt FunctionSpaces.get_constituent_extraction(space, element_id)
        @test_opt FunctionSpaces.get_constituent_polynomial_degree(space)
        @test_opt FunctionSpaces.get_constituent_manifold_indices(space)
        @test_opt FunctionSpaces.get_constituent_element_vertices(space, element_id)
        @test_opt FunctionSpaces.get_constituent_element_lengths(space, element_id)

        if FunctionSpaces.get_manifold_dim(space) == 1
            @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_constituent_local_basis(
                space, element_id, xi_1D, nderivatives
            )
            @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_constituent_evaluations(
                space, element_id, xi_1D, nderivatives
            )
            @test_opt FunctionSpaces.get_constituent_evaluation_points(space, xi_1D)
        elseif FunctionSpaces.get_manifold_dim(space) == 2
            @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_constituent_local_basis(
                space, element_id, xi_2D, nderivatives
            )
            @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_constituent_evaluations(
                space, element_id, xi_2D, nderivatives
            )
            @test_opt FunctionSpaces.get_constituent_evaluation_points(space, xi_2D)
        elseif FunctionSpaces.get_manifold_dim(space) == 3
            @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_constituent_local_basis(
                space, element_id, xi_3D, nderivatives
            )
            @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_constituent_evaluations(
                space, element_id, xi_3D, nderivatives
            )
            @test_opt FunctionSpaces.get_constituent_evaluation_points(space, xi_3D)
        end

    elseif typeof(space) <: FunctionSpaces.DirectSumSpace
        @test_opt FunctionSpaces.get_support(space, basis_id)
        @test_opt FunctionSpaces.get_dof_offsets(space)
        @test_opt FunctionSpaces.get_dof_offsets(space, component_id)
    else
        println("No methods specific to ", typeof(space), ".")
    end

    if FunctionSpaces.get_manifold_dim(space) == 1
        @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_local_basis(
            space, element_id, xi_1D, nderivatives, component_id
        )
        @test_opt ignored_modules = (Memoization,) FunctionSpaces.evaluate(
            space, element_id, xi_1D, nderivatives
        )
        @test_opt ignored_modules = (Memoization,) FunctionSpaces.evaluate(
            space,
            element_id,
            xi_1D,
            nderivatives,
            ones(FunctionSpaces.get_num_basis(space)),
        )
    elseif FunctionSpaces.get_manifold_dim(space) == 2
        @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_local_basis(
            space, element_id, xi_2D, nderivatives, component_id
        )
        @test_opt ignored_modules = (Memoization,) FunctionSpaces.evaluate(
            space, element_id, xi_2D, nderivatives
        )
        @test_opt ignored_modules = (Memoization,) FunctionSpaces.evaluate(
            space,
            element_id,
            xi_2D,
            nderivatives,
            ones(FunctionSpaces.get_num_basis(space)),
        )
    elseif FunctionSpaces.get_manifold_dim(space) == 3
        @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_local_basis(
            space, element_id, xi_3D, nderivatives, component_id
        )
        @test_opt ignored_modules = (Memoization,) FunctionSpaces.evaluate(
            space, element_id, xi_3D, nderivatives
        )
        @test_opt ignored_modules = (Memoization,) FunctionSpaces.evaluate(
            space,
            element_id,
            xi_3D,
            nderivatives,
            ones(FunctionSpaces.get_num_basis(space)),
        )
    else
        @warn "FunctionSpacesInference: This space was not tested: $(space)"
    end
end

end
