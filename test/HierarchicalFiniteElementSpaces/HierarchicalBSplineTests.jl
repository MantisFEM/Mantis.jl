import Mantis

using Test

# Test according to the example in Fig 9. of https://doi.org/10.1007/s11831-022-09752-5

nq = 7
p = 3
nlevels = 3
subdiv = 2

brk = collect(LinRange(0,1,nq))
patch = Mantis.Mesh.Patch1D(brk)
regularity = fill(p-1, nq)
regularity[1]=regularity[nq] = -1
bspline1 = Mantis.FunctionSpaces.BSplineSpace(patch, p, regularity)

two_scale_operators = Vector{Mantis.FunctionSpaces.TwoScaleOperator}(undef, nlevels-1)
bsplines = Vector{Mantis.FunctionSpaces.BSplineSpace}(undef, nlevels)
bsplines[1] = bspline1

for l in 1:nlevels-1
    ts_operator, bspline = Mantis.FunctionSpaces.subdivide_bspline(bsplines[l], subdiv)
    two_scale_operators[l] = ts_operator
    bsplines[l+1] = bspline
end

refined_domains = Mantis.FunctionSpaces.HierarchicalActiveInfo([1,2,3,4,5,6,3,4,5,6,7,8,9,10,7,8,9,10,11,12,13,14,15,16],[0,6,14,24])
hspace = Mantis.FunctionSpaces.get_hierarchical_space(bsplines, two_scale_operators, refined_domains); nothing

# test if active elements are correct   
@test Mantis.FunctionSpaces.get_level_elements(hspace, 1)[2] == [1,6]
@test Mantis.FunctionSpaces.get_level_elements(hspace, 2)[2] == [3,9,10]
@test Mantis.FunctionSpaces.get_level_elements(hspace, 3)[2] == collect(7:16)

# test if active functions are correct   
@test Mantis.FunctionSpaces.get_level_functions(hspace, 1)[2] == [1,2,3,4,6,7,8,9]
@test Mantis.FunctionSpaces.get_level_functions(hspace, 2)[2] == [6,9,10]
@test Mantis.FunctionSpaces.get_level_functions(hspace, 3)[2] == collect(10:16)

# Test if projection in space is exact
nxi = 20
xi = collect(range(0,1, nxi))

xs = Vector{Float64}(undef, Mantis.FunctionSpaces.get_num_elements(hspace)*nxi)
nx = length(xs)

A = zeros(nx, Mantis.FunctionSpaces.get_num_basis(hspace))

for el ∈ 1:1:Mantis.FunctionSpaces.get_num_elements(hspace)
    
    level = Mantis.FunctionSpaces.get_active_level(hspace.active_elements, el)
    element_id = Mantis.FunctionSpaces.get_active_id(hspace.active_elements, el)

    borders = Mantis.Mesh.get_element(hspace.spaces[level].knot_vector.patch_1d, element_id)
    x = borders[1] .+ xi .* (borders[2] - borders[1])

    idx = (el-1)*nxi+1:el*nxi
    xs[idx] = x

    local eval = Mantis.FunctionSpaces.evaluate(hspace, el, (xi,), 0)

    A[idx, eval[2]] = eval[1][1][1]
end

coeffs = A \ xs
A * coeffs .- xs
all(isapprox.(A * coeffs .- xs, 0.0, atol=1e-14))

@test Mantis.FunctionSpaces.get_num_levels(hspace) == nlevels

for l in 1:nlevels
    @test Mantis.FunctionSpaces.get_space(hspace, l) == bsplines[l]
end

# Tests for coefficients and evaluation
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(hspace)

    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(hspace, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity

    # check Hierarchical B-spline evaluation
    h_eval, _ = Mantis.FunctionSpaces.evaluate(hspace, el, (xi,), 0)
    # Positivity of the basis
    @test minimum(h_eval[1][1]) >= 0.0
end