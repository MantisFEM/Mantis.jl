module HierarchicalBSplineTests

using Mantis

using Test

# Test according to the example in Fig 9. of https://doi.org/10.1007/s11831-022-09752-5

nq = 7
p = 3
nlevels = 3
subdiv = 2

brk = collect(LinRange(0, 1, nq))
patch = Geometry.CartesianGeometry(brk)
regularity = fill(p - 1, nq)
regularity[1] = regularity[nq] = -1
bspline1 = FunctionSpaces.BSplineSpace(patch, p, regularity)

two_scale_operators = Vector{FunctionSpaces.TwoScaleOperator}(undef, nlevels - 1)
bsplines = Vector{FunctionSpaces.BSplineSpace}(undef, nlevels)
bsplines[1] = bspline1

for l in 1:(nlevels - 1)
    ts_operator, bspline = FunctionSpaces.build_two_scale_operator(bsplines[l], subdiv)
    two_scale_operators[l] = ts_operator
    bsplines[l + 1] = bspline
end

refined_domains = FunctionSpaces.HierarchicalActiveInfo([
    [1, 2, 3, 4, 5, 6], [3, 4, 5, 6, 7, 8, 9, 10], [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
])
hier_space = FunctionSpaces.HierarchicalFiniteElementSpace(
    bsplines, two_scale_operators, refined_domains, (subdiv,)
)

# test if active elements are correct
@test FunctionSpaces.get_level_element_ids(hier_space, 1) == [1, 6]
@test FunctionSpaces.get_level_element_ids(hier_space, 2) == [3, 9, 10]
@test FunctionSpaces.get_level_element_ids(hier_space, 3) == collect(7:16)

# test if active functions are correct
@test FunctionSpaces.get_level_basis_ids(hier_space, 1) == [1, 2, 3, 4, 6, 7, 8, 9]
@test FunctionSpaces.get_level_basis_ids(hier_space, 2) == [6, 9, 10]
@test FunctionSpaces.get_level_basis_ids(hier_space, 3) == collect(10:16)

# Test if projection in space is exact
nxi = 20
xi = Points.CartesianPoints((range(0, 1, nxi),))

xs = Vector{Float64}(undef, FunctionSpaces.get_num_elements(hier_space) * nxi)
nx = length(xs)

A = zeros(nx, FunctionSpaces.get_num_basis(hier_space))

for element_id in 1:1:FunctionSpaces.get_num_elements(hier_space)
    level, element_level_id = FunctionSpaces.convert_to_element_level_and_level_id(
        hier_space, element_id
    )
	borders = FunctionSpaces.get_element_vertices(hier_space, element_id)[1]
    x = borders[1] .+ Points.get_constituent_points(xi)[1] .* (borders[2] - borders[1])
    idx = ((element_id - 1) * nxi + 1):(element_id * nxi)
    xs[idx] = x
    h_eval, h_inds = FunctionSpaces.evaluate(hier_space, element_id, xi, 0)

    A[idx, h_inds] = h_eval[1][1][1]
end

coeffs = A \ xs
A * coeffs .- xs
all(isapprox.(A * coeffs .- xs, 0.0, atol=1e-14))

@test FunctionSpaces.get_num_levels(hier_space) == nlevels

for l in 1:nlevels
    @test FunctionSpaces.get_space(hier_space, l) == bsplines[l]
end

# Tests for coefficients and evaluation
for element_id in 1:1:FunctionSpaces.get_num_elements(hier_space)

    # check extraction coefficients
    ex_coeffs, _ = FunctionSpaces.get_extraction(hier_space, element_id)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity

    # check Hierarchical B-spline evaluation
    h_eval, _ = FunctionSpaces.evaluate(hier_space, element_id, xi, 0)
    # Positivity of the basis
    @test minimum(h_eval[1][1][1]) >= 0.0
end

end
