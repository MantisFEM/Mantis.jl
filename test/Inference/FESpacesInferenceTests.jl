module FESpacesInferenceTests

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
nodes = Points.get_input_points(Quadrature.get_nodes(Quadrature.gauss_lobatto(2)))[1]
nodes2 = Points.get_input_points(Quadrature.get_nodes(Quadrature.gauss_legendre(2)))[1]
ll_poly = FunctionSpaces.Lagrange(nodes)
gl_poly = FunctionSpaces.Lagrange(nodes2)
el_poly = FunctionSpaces.Edge(nodes)
B1LL = FunctionSpaces.BSplineSpace(geometry1, geometry1multi, ll_poly, fill(-1, 26))
B1GL = FunctionSpaces.BSplineSpace(geometry1, geometry1multi, gl_poly, fill(-1, 26))
B1ELL = FunctionSpaces.BSplineSpace(geometry1, geometry1multi, el_poly, fill(-1, 26))

# Rational
R1 = FunctionSpaces.RationalFESpace(B1, [0.2, 0.8])
R1m = FunctionSpaces.RationalFESpace(B1multi, rand(FunctionSpaces.get_num_basis(B1multi)))

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

# The helper function used here creates a TensorProductSpace using a CartesianGeometry. The
# above constructors use a TensorProductGeometry instead.
B1_3D = FunctionSpaces.create_bspline_space(
    (0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0),
    (3, 4, 5),
    (FunctionSpaces.Bernstein(2), FunctionSpaces.Bernstein(2), FunctionSpaces.Bernstein(3)),
    (1, 1, 1),
)

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
DS_TPB1mTPR1mTPBR1m = FunctionSpaces.DirectSumSpace((TP_B1mB1m, TP_R1mR1m, TP_B1mR1m))

# Hierarchical
ref_space_h2(s) = FunctionSpaces.refinement_uniform(s, 2)
ref_geo_h2(s) = Geometry.refinement_uniform(s, 2)
ref_space_p1(s) = FunctionSpaces.refinement_degree(s, 1)
scal_space_p1(p, c) = FunctionSpaces.scaling_matrix_degree(p, c, 1)
scal_space_h2 = FunctionSpaces.scaling_matrix_uniform
SelStd = FunctionSpaces.SelectionStandard()
SelSimp = FunctionSpaces.SelectionSimple()
HB = FunctionSpaces.HB()
THB = FunctionSpaces.THB()

geo_l1 = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 6),))
geo_l2 = Hierarchical.Refinement(geo_l1, ref_geo_h2)()
geo_l3 = Hierarchical.Refinement(geo_l2, ref_geo_h2)()
geo_scal_1 = Geometry.scaling_uniform(geo_l1, geo_l2, 2)
geo_scal_2 = Geometry.scaling_uniform(geo_l2, geo_l3, 2)

active = Hierarchical.ActiveInfo([[1, 2], [7, 8, 9, 10], [9, 10, 11, 12]])
hgeo = Geometry.HierarchicalGeometry(
    Hierarchical.NestedHierarchy(active, geo_scal_1, geo_scal_2)
)

bsp_l1 = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (5,), (1,), (0,))
bsp_l12 = Hierarchical.Refinement(bsp_l1, ref_space_h2)()
bsp_l2 = Hierarchical.Refinement(bsp_l12, ref_space_p1)()
bsp_l22 = Hierarchical.Refinement(bsp_l2, ref_space_h2)()
bsp_l3 = Hierarchical.Refinement(bsp_l22, ref_space_p1)()
bsp_scal_1 = Hierarchical.MatrixScaling(
    (bsp_l1, bsp_l12, bsp_l2), scal_space_h2, scal_space_p1
)
bsp_scal_2 = Hierarchical.MatrixScaling(
    (bsp_l2, bsp_l22, bsp_l3), scal_space_h2, scal_space_p1
)
scalings = (bsp_scal_1, bsp_scal_2)
HB_Std1 = FunctionSpaces.HierarchicalSpace(hgeo, hgeo, scalings, SelStd, HB)
HB_Simp1 = FunctionSpaces.HierarchicalSpace(hgeo, hgeo, scalings, SelSimp, HB)
THB_Std1 = FunctionSpaces.HierarchicalSpace(hgeo, hgeo, scalings, SelStd, THB)
THB_Simp1 = FunctionSpaces.HierarchicalSpace(hgeo, hgeo, scalings, SelSimp, THB)

spaces = (
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
    B1_3D,
    DS1,
    DS_B1mB1m,
    DS_TP1mTP1m,
    DS_TP1mTPR1mTP1m,
    DS_TPB1mTPR1mTPBR1m,
    HB_Std1,
    HB_Simp1,
    THB_Std1,
    THB_Simp1,
)

# Note that JET only uses the types of the inputs, so which numbers we pick
# here is irrelevant.
xi_1D = Points.TensorProductPoints(([0.0, 1.0],))
xi_2D = Points.TensorProductPoints(([0.0, 1.0], [0.0, 1.0]))
xi_3D = Points.TensorProductPoints(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))

element_id = 1
component_id = 1
nderivatives = 2
basis_id = 4

for space in spaces

    # Methods on the type.
    @test_opt FunctionSpaces.get_manifold_dim(space)
    @test_opt FunctionSpaces.get_num_components(space)
    @test_opt FunctionSpaces.get_num_patches(space)

    # Methods which have a general fallback (can be an error fallback).
    @test_opt FunctionSpaces.get_component_spaces(space)
    @test_opt FunctionSpaces.get_extraction_operator(space)
    @test_opt FunctionSpaces.get_extraction(space, element_id, component_id)
    @test_opt FunctionSpaces.get_extraction_coefficients(space, element_id, component_id)
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
        @test_opt FunctionSpaces.get_factor_spaces(space)
        @test_opt FunctionSpaces.get_cart_num_elements(space)
        @test_opt FunctionSpaces.get_lin_num_elements(space)
        @test_opt FunctionSpaces.get_num_spaces(space)
        @test_opt FunctionSpaces.get_support(space, basis_id)

        @test_opt FunctionSpaces.get_factor_element_ids(space, element_id)
        @test_opt FunctionSpaces.get_factor_basis_ids(space, basis_id)
        @test_opt FunctionSpaces.get_factor_num_basis(space)
        @test_opt FunctionSpaces.get_factor_num_basis(space, element_id)
        @test_opt FunctionSpaces.get_factor_manifold_dims(space)
        @test_opt FunctionSpaces.get_factor_basis_indices(space, element_id)
        @test_opt FunctionSpaces.get_factor_supports(space, element_id)
        @test_opt FunctionSpaces.get_factor_extractions(space, element_id)
        @test_opt FunctionSpaces.get_factor_polynomial_degrees(space)
        @test_opt FunctionSpaces.get_factor_manifold_indices(space)
        @test_opt FunctionSpaces.get_factor_element_vertices(space, element_id)
        @test_opt FunctionSpaces.get_factor_element_lengths(space, element_id)

        if FunctionSpaces.get_manifold_dim(space) == 1
            @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_factor_local_basis(
                space, element_id, xi_1D, nderivatives
            )
            @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_factor_evaluations(
                space, element_id, xi_1D, nderivatives
            )
            @test_opt FunctionSpaces.get_factor_evaluation_points(space, xi_1D)
        elseif FunctionSpaces.get_manifold_dim(space) == 2
            @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_factor_local_basis(
                space, element_id, xi_2D, nderivatives
            )
            @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_factor_evaluations(
                space, element_id, xi_2D, nderivatives
            )
            @test_opt FunctionSpaces.get_factor_evaluation_points(space, xi_2D)
        elseif FunctionSpaces.get_manifold_dim(space) == 3
            @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_factor_local_basis(
                space, element_id, xi_3D, nderivatives
            )
            @test_opt ignored_modules = (Memoization,) FunctionSpaces.get_factor_evaluations(
                space, element_id, xi_3D, nderivatives
            )
            @test_opt FunctionSpaces.get_factor_evaluation_points(space, xi_3D)
        end

    elseif typeof(space) <: FunctionSpaces.DirectSumSpace
        @test_opt FunctionSpaces.get_support(space, basis_id)
        @test_opt FunctionSpaces.get_dof_offsets(space)
        @test_opt FunctionSpaces.get_dof_offsets(space, component_id)
    elseif typeof(space) <: FunctionSpaces.HierarchicalSpace
        @test_opt FunctionSpaces.get_support(space, element_id)
        @test_call FunctionSpaces.get_support(space, element_id)
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
