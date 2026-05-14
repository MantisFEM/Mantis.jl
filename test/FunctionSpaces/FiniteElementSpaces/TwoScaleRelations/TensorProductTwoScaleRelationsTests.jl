module TensorProductTwoScaleRelationsTests

using Mantis

using Test

# Test parameters
breakpoints1 = [0.0, 0.5, 1.0]
patch1 = Geometry.CartesianGeometry(breakpoints1)
breakpoints2 = [0.0, 0.5, 0.6, 1.0]
patch2 = Geometry.CartesianGeometry(breakpoints2)
breakpoints3 = [0.0, 1.0]
patch3 = Geometry.CartesianGeometry(breakpoints3)

deg1 = 2
deg2 = 3
deg3 = 1

B1 = FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1 - 1, -1])
B2 = FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2 - 1, 1), deg2 - 1, -1])
B3 = FunctionSpaces.BSplineSpace(patch3, deg3, [-1, -1])

nsub1 = (3,)
nsub2 = (2,)
nsub3 = (4,)
RS1 = FunctionSpaces.GlobalRefinement(nsub1, (0,), (deg1-1,), :h_first)
RS2 = FunctionSpaces.GlobalRefinement(nsub2, (0,), (deg2-1,), :h_first)
RS3 = FunctionSpaces.GlobalRefinement(nsub3, (0,), (deg3-1,), :h_first)

ts1, BF1 = FunctionSpaces.build_two_scale_operator(B1, RS1)
ts2, BF2 = FunctionSpaces.build_two_scale_operator(B2, RS2)
ts3, BF3 = FunctionSpaces.build_two_scale_operator(B3, RS3)

parent_space_1 = FunctionSpaces.TensorProductSpace((
    FunctionSpaces.TensorProductSpace((B1, B2)), B3
))
child_space_1 = FunctionSpaces.TensorProductSpace((
    FunctionSpaces.TensorProductSpace((BF1, BF2)), BF3
))
parent_space_2 = FunctionSpaces.TensorProductSpace((
    B1, FunctionSpaces.TensorProductSpace((B2, B3))
))
child_space_2 = FunctionSpaces.TensorProductSpace((
    BF1, FunctionSpaces.TensorProductSpace((BF2, BF3))
))
parent_space_3 = FunctionSpaces.TensorProductSpace((B1, B2, B3))
child_space_3 = FunctionSpaces.TensorProductSpace((BF1, BF2, BF3))
ts_operators_1 = FunctionSpaces.TensorProductTwoScaleOperator(
    FunctionSpaces.TensorProductSpace((B1, B2)),
    FunctionSpaces.TensorProductSpace((BF1, BF2)),
    (ts1, ts2),
)
ts_operators_2 = FunctionSpaces.TensorProductTwoScaleOperator(
    FunctionSpaces.TensorProductSpace((B2, B3)),
    FunctionSpaces.TensorProductSpace((BF2, BF3)),
    (ts2, ts3),
)
ts_operators_3 = (ts1, ts2, ts3)

tts_1 = FunctionSpaces.TensorProductTwoScaleOperator(
    parent_space_1, child_space_1, (ts_operators_1, ts3)
)
tts_2 = FunctionSpaces.TensorProductTwoScaleOperator(
    parent_space_2, child_space_2, (ts1, ts_operators_2)
)
tts_3 = FunctionSpaces.TensorProductTwoScaleOperator(
    parent_space_3, child_space_3, ts_operators_3
)

# Tests for TensorTwoScaleOperator

# Check if global subdivision matrices are equal
@test isapprox(
    FunctionSpaces.get_global_subdiv_matrix(tts_1),
    FunctionSpaces.get_global_subdiv_matrix(tts_2),
    atol=1e-15,
)
@test isapprox(
    FunctionSpaces.get_global_subdiv_matrix(tts_2),
    FunctionSpaces.get_global_subdiv_matrix(tts_3),
    atol=1e-15,
)
# Check total number of parent elements
@test FunctionSpaces.get_num_elements(FunctionSpaces.get_parent_space(tts_1)) ==
    FunctionSpaces.get_num_elements(FunctionSpaces.get_parent_space(tts_2))
@test FunctionSpaces.get_num_elements(FunctionSpaces.get_parent_space(tts_2)) ==
    FunctionSpaces.get_num_elements(FunctionSpaces.get_parent_space(tts_3))
@test FunctionSpaces.get_num_elements(FunctionSpaces.get_child_space(tts_1)) ==
    FunctionSpaces.get_num_elements(FunctionSpaces.get_child_space(tts_2))
@test FunctionSpaces.get_num_elements(FunctionSpaces.get_child_space(tts_2)) ==
    FunctionSpaces.get_num_elements(FunctionSpaces.get_child_space(tts_3))

# Check if local subdivision matrices are the same.
for el in 1:FunctionSpaces.get_num_elements(FunctionSpaces.get_child_space(tts_1))
    # Check if parent elements are the same
    parent_el_1 = FunctionSpaces.get_element_parent(tts_1, el)
    parent_el_2 = FunctionSpaces.get_element_parent(tts_2, el)
    parent_el_3 = FunctionSpaces.get_element_parent(tts_3, el)
    @test parent_el_1 == parent_el_2
    @test parent_el_2 == parent_el_3
    @test isapprox(
        FunctionSpaces.get_local_subdiv_matrix(tts_1, parent_el_1, el),
        FunctionSpaces.get_local_subdiv_matrix(tts_2, parent_el_2, el),
        atol=1e-15,
    )
    @test isapprox(
        FunctionSpaces.get_local_subdiv_matrix(tts_2, parent_el_2, el),
        FunctionSpaces.get_local_subdiv_matrix(tts_3, parent_el_3, el),
        atol=1e-15,
    )
end

# Check if child elements are the same.
for el in 1:1:FunctionSpaces.get_num_elements(FunctionSpaces.get_parent_space(tts_1))
    @test FunctionSpaces.get_element_children(tts_1, el) ==
        FunctionSpaces.get_element_children(tts_2, el)
    @test FunctionSpaces.get_element_children(tts_2, el) ==
        FunctionSpaces.get_element_children(tts_3, el)
end

# Check if total number of basis functions are the same
@test FunctionSpaces.get_num_basis(FunctionSpaces.get_parent_space(tts_1)) ==
    FunctionSpaces.get_num_basis(FunctionSpaces.get_parent_space(tts_2))
@test FunctionSpaces.get_num_basis(FunctionSpaces.get_parent_space(tts_2)) ==
    FunctionSpaces.get_num_basis(FunctionSpaces.get_parent_space(tts_3))
@test FunctionSpaces.get_num_basis(FunctionSpaces.get_child_space(tts_1)) ==
    FunctionSpaces.get_num_basis(FunctionSpaces.get_child_space(tts_2))
@test FunctionSpaces.get_num_basis(FunctionSpaces.get_child_space(tts_2)) ==
    FunctionSpaces.get_num_basis(FunctionSpaces.get_child_space(tts_3))

# Check if basis parents are the same
for basis_id in 1:FunctionSpaces.get_num_basis(FunctionSpaces.get_child_space(tts_1))
    @test FunctionSpaces.get_basis_parents(tts_1, basis_id) ==
        FunctionSpaces.get_basis_parents(tts_2, basis_id)
    @test FunctionSpaces.get_basis_parents(tts_2, basis_id) ==
        FunctionSpaces.get_basis_parents(tts_3, basis_id)
end

# Check if basis children are the same
for basis_id in 1:FunctionSpaces.get_num_basis(FunctionSpaces.get_parent_space(tts_1))
    @test FunctionSpaces.get_basis_children(tts_1, basis_id) ==
        FunctionSpaces.get_basis_children(tts_2, basis_id)
    @test FunctionSpaces.get_basis_children(tts_2, basis_id) ==
        FunctionSpaces.get_basis_children(tts_3, basis_id)
end

end
