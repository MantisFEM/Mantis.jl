module TensorProductTHBSplineTests

using Mantis

using Test

# Tests for a tensor product HierarchicalSplineSpace
ne1 = 5
ne2 = 5
breakpoints1 = collect(range(0, 1, ne1 + 1))
patch1 = Geometry.CartesianGeometry(breakpoints1)
breakpoints2 = collect(range(0, 1, ne2 + 1))
patch2 = Geometry.CartesianGeometry(breakpoints2)
deg1 = 2
deg2 = 2
nsubs = (3, 3)
nlevels = 3

CB1 = FunctionSpaces.BSplineSpace(patch1, deg1, [-1; fill(deg1 - 1, ne1 - 1); -1])
CB2 = FunctionSpaces.BSplineSpace(patch2, deg2, [-1; fill(deg2 - 1, ne2 - 1); -1])
CTP = FunctionSpaces.TensorProductSpace((CB1, CB2))

TTS, FTP = FunctionSpaces.build_two_scale_operator(CTP, nsubs)

spaces = [CTP, FTP]
operators = [TTS]

for level in 3:nlevels
    new_operator, new_space = FunctionSpaces.build_two_scale_operator(
        spaces[level - 1], nsubs
    )
    push!(spaces, new_space)
    push!(operators, new_operator)
end

level_2_marked_elements = [
    child for parent in [7, 8, 9, 12, 13, 14, 17, 18, 19] for
    child in FunctionSpaces.get_element_children(operators[1], parent)
]
level_3_marked_elements = [
    child for parent in [23, 24, 25, 33, 34, 35, 43, 44, 45] for
    child in FunctionSpaces.get_element_children(operators[2], parent)
]

marked_elements_per_level = [Int[], level_2_marked_elements, level_3_marked_elements]
hier_space = FunctionSpaces.HierarchicalFiniteElementSpace(
    spaces, operators, marked_elements_per_level, nsubs, true
)

qrule = Quadrature.tensor_product_rule((deg1 + 1, deg2 + 1), Quadrature.gauss_legendre)
xi = Quadrature.get_nodes(qrule)

# Tests for coefficients and evaluation
for el in 1:1:FunctionSpaces.get_num_elements(hier_space)
    # check extraction coefficients
    ex_coeffs, _ = FunctionSpaces.get_extraction(hier_space, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity

    # check Hierarchical B-spline evaluation
    h_eval, _ = FunctionSpaces.evaluate(hier_space, el, xi)
    # Positivity of the basis
    @test minimum(h_eval[1][1][1]) >= 0.0
    # Partition of unity
    @test all(isapprox.(sum(h_eval[1][1][1]; dims=2), 1.0, atol=1e-14))
end

end
