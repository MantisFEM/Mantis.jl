module FormEvaluationTests

using Mantis

using Test

# 2D tests --------------------------------------------------------------------

# Setup the geometry

# First the 2D rectangular domain parameters
starting_point_2d = (0.0, 0.0)
box_size_2d = (1.0, 1.0)
num_elements_2d = (2, 3)
degree_2d = (2, 2)
regularity_2d = degree_2d .- 1

geo_2d_cart = Geometry.create_cartesian_box(starting_point_2d, box_size_2d, num_elements_2d)

# Setup the form spaces
TP_Space_2d = FunctionSpaces.create_bspline_space(
    starting_point_2d, box_size_2d, num_elements_2d, degree_2d, regularity_2d
)

# Generate the multicomponent FEMSpace for the 1-form.
dsTP_1_form_2d = FunctionSpaces.DirectSumSpace((TP_Space_2d, TP_Space_2d))

# Generate the form spaces
zero_form_space_2d = Forms.FormSpace(0, TP_Space_2d, "ν")
one_form_space_2d = Forms.FormSpace(1, dsTP_1_form_2d, "η")
top_form_space_2d = Forms.FormSpace(2, TP_Space_2d, "σ")

# Checks if incorrect arguments throw errors
@test_throws ArgumentError Forms.FormSpace(0, dsTP_1_form_2d, "ν")
@test_throws ArgumentError Forms.FormSpace(1, TP_Space_2d, "ν")
@test_throws ArgumentError Forms.FormSpace(2, dsTP_1_form_2d, "ν")

# Evaluate forms
# Evaluation points
ξ_1_evaluation = [0.0, 1.0]
ξ_2_evaluation = [0.0, 1.0]
ξ_evaluation = Points.TensorProductPoints((ξ_1_evaluation, ξ_2_evaluation))
n_evaluation_points = Points.get_num_points(ξ_evaluation)

# Element where to evaluate
element_idx = 1

# Global basis indices of component `component_id` of `space` on the element.
function get_component_indices(space, component_id)
    return collect(FunctionSpaces.get_basis_indices(space, element_idx, component_id))
end

# 0-forms
zero_form_space_eval, zero_form_space_idx = Forms.evaluate(
    zero_form_space_2d, element_idx, ξ_evaluation
)

# Test output
n_active_basis_0_form = FunctionSpaces.get_num_basis(TP_Space_2d, element_idx)

# Single component output
@test size(zero_form_space_eval) == (1,)
# Check that we get a matrix with the correct dimensions for the single component.
@test size(zero_form_space_eval[1]) == (n_evaluation_points, n_active_basis_0_form)
# Single component with a single expression with the active basis in the element.
@test zero_form_space_idx == [[get_component_indices(TP_Space_2d, 1)]]

# 1-forms
one_form_space_eval, one_form_space_idx = Forms.evaluate(
    one_form_space_2d, element_idx, ξ_evaluation
)

# Test output
n_active_basis_1_form = FunctionSpaces.get_num_basis(dsTP_1_form_2d, element_idx)
n_nonzero_basis_in_xi_1 = FunctionSpaces.get_num_basis(TP_Space_2d, element_idx)
n_nonzero_basis_in_xi_2 = n_nonzero_basis_in_xi_1  # Because the component spaces are equal.

@test size(one_form_space_eval) == (2,)  # check that we get two components
# Each component is only evaluated on the basis functions of its own component space,
# because the underlying space is a direct sum space.
@test size(one_form_space_eval[1]) == (n_evaluation_points, n_nonzero_basis_in_xi_1)
@test size(one_form_space_eval[2]) == (n_evaluation_points, n_nonzero_basis_in_xi_2)
@test one_form_space_idx == [
    [get_component_indices(dsTP_1_form_2d, 1)], [get_component_indices(dsTP_1_form_2d, 2)]
]
# Together, the components contain all the active basis in the element.
@test sort(vcat(one_form_space_idx[1][1], one_form_space_idx[2][1])) ==
    sort(collect(FunctionSpaces.get_basis_indices(dsTP_1_form_2d, element_idx)))
@test length(one_form_space_idx[1][1]) + length(one_form_space_idx[2][1]) ==
    n_active_basis_1_form

# top-forms
top_form_space_eval, top_form_space_idx = Forms.evaluate(
    top_form_space_2d, element_idx, ξ_evaluation
)

# Test output
n_active_basis_top_form = FunctionSpaces.get_num_basis(TP_Space_2d, element_idx)

@test size(top_form_space_eval) == (1,)  # check that we get one component
# check that we get a matrix with the correct dimensions for the single component
@test size(top_form_space_eval[1]) == (n_evaluation_points, n_active_basis_top_form)
# check that we get one single vector with as many indices as the active basis in the element
@test top_form_space_idx == [[get_component_indices(TP_Space_2d, 1)]]

# Exterior derivative

# 0-forms
d_zero_form_space_2d = Forms.ExteriorDerivative(zero_form_space_2d)
d_zero_form_space_eval, d_zero_form_space_idx = Forms.evaluate(
    d_zero_form_space_2d, element_idx, ξ_evaluation
)

# Test output
n_active_basis_0_form = FunctionSpaces.get_num_basis(TP_Space_2d, element_idx)

@test size(d_zero_form_space_eval) == (2,)  # check that we get two components (2D)
# All components of the gradient are spanned by the basis of the 0-form.
@test d_zero_form_space_idx == [[get_component_indices(TP_Space_2d, 1)] for _ in 1:2]
@test all([
    size(form_component) == (n_evaluation_points, n_active_basis_0_form) for
    form_component in d_zero_form_space_eval
])  # check that we get a matrix with the correct dimensions per component
@test isapprox(sum(sum.(d_zero_form_space_eval)), 0.0, atol=1e-12)

# 1-forms
d_one_form_space_2d = Forms.ExteriorDerivative(one_form_space_2d)
d_one_form_space_eval, d_one_form_space_idx = Forms.evaluate(
    d_one_form_space_2d, element_idx, ξ_evaluation
)

# Test output
n_active_basis_1_form = FunctionSpaces.get_num_basis(dsTP_1_form_2d, element_idx)

@test size(d_one_form_space_eval) == (1,)  # check that we get one component (2D)
# The curl mixes both components, so it is evaluated on the union of their indices.
@test d_one_form_space_idx == [[
    sort(
        union(
            get_component_indices(dsTP_1_form_2d, 1),
            get_component_indices(dsTP_1_form_2d, 2),
        ),
    ),
]]
@test all([
    size(form_component) == (n_evaluation_points, n_active_basis_1_form) for
    form_component in d_one_form_space_eval
])  # check that we get a matrix with the correct dimensions per component
@test isapprox(sum(sum.(d_zero_form_space_eval)), 0.0, atol=1e-12)

# top-forms
@test_throws ArgumentError Forms.ExteriorDerivative(top_form_space_2d)

# hodge_zero_form_space_eval, hodge_zero_form_space_idx = Forms.evaluate_hodge_star(zero_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))

# d_one_form_space_eval, d_one_form_space_idx = Forms.evaluate(one_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))
# hodge_one_form_space_eval, hodge_zero_form_space_idx = Forms.evaluate_hodge_star(one_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))

# top_form_space_eval, top_form_space_idx = Forms.evaluate(top_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))
# hodge_top_form_space_eval, hodge_top_form_space_idx = Forms.evaluate_hodge_star(top_form_space, 1, ([0.0, 1.0], [0.0, 1.0]))

# -----------------------------------------------------------------------------

# 3D tests --------------------------------------------------------------------

# Setup the geometry

# First the 3D rectangular cuboid domain parameters
starting_point_3d = (0.0, 0.0, 0.0)
box_size_3d = (1.0, 1.0, 1.0)
num_elements_3d = (2, 3, 2)
degree_3d = (2, 2, 3)
regularity_3d = degree_3d .- 1

# Generate the 3D rectangular cuboid straight geometry
geo_3d_cart = Geometry.create_cartesian_box(starting_point_3d, box_size_3d, num_elements_3d)

# Setup the form spaces

# Generate a tensor product space (single valued space)
TP_Space_3d = FunctionSpaces.create_bspline_space(
    starting_point_3d, box_size_3d, num_elements_3d, degree_3d, regularity_3d
)

# Generate the multivalued FEMSpaces (DirectSumSpace)
dsTP_0_form_3d = FunctionSpaces.DirectSumSpace((TP_Space_3d,))  # direct sum space
dsTP_1_form_3d = FunctionSpaces.DirectSumSpace((TP_Space_3d, TP_Space_3d, TP_Space_3d))  # direct sum space
dsTP_2_form_3d = dsTP_1_form_3d
dsTP_top_form_3d = dsTP_0_form_3d

# Generate the form spaces
zero_form_space_3d = Forms.FormSpace(0, dsTP_0_form_3d, "ν")
one_form_space_3d = Forms.FormSpace(1, dsTP_1_form_3d, "η")
two_form_space_3d = Forms.FormSpace(2, dsTP_2_form_3d, "ψ")
top_form_space_3d = Forms.FormSpace(3, dsTP_top_form_3d, "σ")

# Checks if incorrect arguments throw errors
@test_throws ArgumentError Forms.FormSpace(1, dsTP_0_form_3d, "ν")
@test_throws ArgumentError Forms.FormSpace(2, dsTP_0_form_3d, "ν")
@test_throws ArgumentError Forms.FormSpace(0, dsTP_1_form_3d, "ν")
@test_throws ArgumentError Forms.FormSpace(3, dsTP_1_form_3d, "ν")

# Evaluate forms

# Evaluation points
ξ_1_evaluation = [0.0, 1.0]
ξ_2_evaluation = [0.0, 1.0]
ξ_3_evaluation = [0.0, 1.0]
ξ_evaluation = Points.TensorProductPoints((ξ_1_evaluation, ξ_2_evaluation, ξ_3_evaluation))
n_evaluation_points = Points.get_num_points(ξ_evaluation)

# Element where to evaluate
element_idx = 1

# 0-forms
zero_form_space_eval, zero_form_space_idx = Forms.evaluate(
    zero_form_space_3d, element_idx, ξ_evaluation
)

# Test output
n_active_basis_0_form = FunctionSpaces.get_num_basis(dsTP_0_form_3d, element_idx)

@test size(zero_form_space_eval) == (1,)  # check that we get one components
@test size(zero_form_space_eval[1]) == (n_evaluation_points, n_active_basis_0_form)   # check that we get a matrix with the correct dimensions for the single component
@test zero_form_space_idx == [[get_component_indices(dsTP_0_form_3d, 1)]]

# 1-forms
one_form_space_eval, one_form_space_idx = Forms.evaluate(
    one_form_space_3d, element_idx, ξ_evaluation
)

# Test output
n_active_basis_1_form = FunctionSpaces.get_num_basis(dsTP_1_form_3d, element_idx)
n_nonzero_basis_in_xi_1 = FunctionSpaces.get_num_basis(TP_Space_3d, element_idx)
n_nonzero_basis_in_xi_2 = n_nonzero_basis_in_xi_1  # because we are using the same spaces in the direct sum
n_nonzero_basis_in_xi_3 = n_nonzero_basis_in_xi_1  # because we are using the same spaces in the direct sum

@test size(one_form_space_eval) == (3,)  # check that we get three components (3D)
# Each component is only evaluated on the basis functions of its own component space,
# because the underlying space is a direct sum space.
@test all([
    size(form_component) == (n_evaluation_points, n_nonzero_basis_in_xi_1) for
    form_component in one_form_space_eval
])
@test one_form_space_idx ==
    [[get_component_indices(dsTP_1_form_3d, component_id)] for component_id in 1:3]
@test sort(reduce(vcat, (only(idx) for idx in one_form_space_idx))) ==
    sort(collect(FunctionSpaces.get_basis_indices(dsTP_1_form_3d, element_idx)))
@test sum(length ∘ only, one_form_space_idx) == n_active_basis_1_form

# 2-forms
two_form_space_eval, two_form_space_idx = Forms.evaluate(
    two_form_space_3d, element_idx, ξ_evaluation
)

# Test output
n_active_basis_2_form = FunctionSpaces.get_num_basis(dsTP_2_form_3d, element_idx)
n_nonzero_basis_in_xi_1 = FunctionSpaces.get_num_basis(TP_Space_3d, element_idx)
n_nonzero_basis_in_xi_2 = n_nonzero_basis_in_xi_1  # because we are using the same spaces in the direct sum
n_nonzero_basis_in_xi_3 = n_nonzero_basis_in_xi_1  # because we are using the same spaces in the direct sum

@test size(two_form_space_eval) == (3,)  # check that we get three components (3D)
# Each component is only evaluated on the basis functions of its own component space,
# because the underlying space is a direct sum space.
@test all([
    size(form_component) == (n_evaluation_points, n_nonzero_basis_in_xi_1) for
    form_component in two_form_space_eval
])
@test two_form_space_idx ==
    [[get_component_indices(dsTP_2_form_3d, component_id)] for component_id in 1:3]
@test sum(length ∘ only, two_form_space_idx) == n_active_basis_2_form

# top-forms
top_form_space_eval, top_form_space_idx = Forms.evaluate(
    top_form_space_3d, element_idx, ξ_evaluation
)

# Test output
n_active_basis_top_form = FunctionSpaces.get_num_basis(dsTP_top_form_3d, element_idx)

@test size(top_form_space_eval) == (1,)  # check that we get one components
@test size(top_form_space_eval[1]) == (n_evaluation_points, n_active_basis_top_form)   # check that we get a matrix with the correct dimensions for the single component
@test top_form_space_idx == [[get_component_indices(dsTP_top_form_3d, 1)]]

# Exterior derivative

# 0-forms
d_zero_form_space_3d = Forms.ExteriorDerivative(zero_form_space_3d)
d_zero_form_space_eval, d_zero_form_space_idx = Forms.evaluate(
    d_zero_form_space_3d, element_idx, ξ_evaluation
)

# Test output
n_active_basis_0_form = FunctionSpaces.get_num_basis(dsTP_0_form_3d, element_idx)

@test size(d_zero_form_space_eval) == (3,)  # check that we get three components (3D)
# All components of the gradient are spanned by the basis of the 0-form.
@test d_zero_form_space_idx == [[get_component_indices(dsTP_0_form_3d, 1)] for _ in 1:3]
@test all([
    size(form_component) == (n_evaluation_points, n_active_basis_0_form) for
    form_component in d_zero_form_space_eval
])  # check that we get a matrix with the correct dimensions per component
@test isapprox(sum(sum.(d_zero_form_space_eval)), 0.0, atol=1e-12)

# 1-forms
d_one_form_space_3d = Forms.ExteriorDerivative(one_form_space_3d)
d_one_form_space_eval, d_one_form_space_idx = Forms.evaluate(
    d_one_form_space_3d, element_idx, ξ_evaluation
)

# Test output
n_active_basis_1_form = FunctionSpaces.get_num_basis(dsTP_1_form_3d, element_idx)

@test size(d_one_form_space_eval) == (3,)  # check that we get three components (3D)
# Component i of the curl mixes components j and k of the 1-form, with (i, j, k) a cyclic
# permutation of (1, 2, 3), so it is evaluated on the union of their indices.
for (i, j, k) in ((1, 2, 3), (2, 3, 1), (3, 1, 2))
    merged_indices = sort(
        union(
            get_component_indices(dsTP_1_form_3d, j),
            get_component_indices(dsTP_1_form_3d, k),
        ),
    )
    @test d_one_form_space_idx[i] == [merged_indices]
    @test size(d_one_form_space_eval[i]) == (n_evaluation_points, length(merged_indices))
end
@test isapprox(sum(sum.(d_zero_form_space_eval)), 0.0, atol=1e-12)

# 2-forms
d_two_form_space_3d = Forms.ExteriorDerivative(two_form_space_3d)
d_two_form_space_eval, d_two_form_space_idx = Forms.evaluate(
    d_two_form_space_3d, element_idx, ξ_evaluation
)

# Test output
n_active_basis_two_form = FunctionSpaces.get_num_basis(dsTP_2_form_3d, element_idx)

@test size(d_two_form_space_eval) == (1,)  # check that we get one component (3D)
# The divergence mixes all components, so it is evaluated on all active basis functions.
@test d_two_form_space_idx ==
    [[sort(collect(FunctionSpaces.get_basis_indices(dsTP_2_form_3d, element_idx)))]]
@test all([
    size(form_component) == (n_evaluation_points, n_active_basis_2_form) for
    form_component in d_two_form_space_eval
])  # check that we get a matrix with the correct dimensions per component
@test isapprox(sum(sum.(d_two_form_space_eval)), 0.0, atol=1e-12)

# top-forms
@test_throws ArgumentError Forms.ExteriorDerivative(top_form_space_3d)

end
