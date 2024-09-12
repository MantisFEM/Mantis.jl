import Mantis

using Test

# Tests for a tensor product HierarchicalSplineSpace

ne1 = 5
ne2 = 5
breakpoints1 = collect(range(0,1,ne1+1))
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = collect(range(0,1,ne2+1))
patch2 = Mantis.Mesh.Patch1D(breakpoints2)

deg1 = 2
deg2 = 2

CB1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1; fill(deg1-1, ne1-1); -1])
CB2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1; fill(deg2-1, ne2-1); -1])

nsub1 = 2
nsub2 = 2

TS1,FB1 = Mantis.FunctionSpaces.subdivide_bspline(CB1, nsub1)
TS2, FB2 = Mantis.FunctionSpaces.subdivide_bspline(CB2, nsub2)

CTP = Mantis.FunctionSpaces.TensorProductSpace(CB1, CB2)
FTP = Mantis.FunctionSpaces.TensorProductSpace(FB1, FB2)
spaces = [CTP, FTP]

CTP_num_els = Mantis.FunctionSpaces.get_num_elements(CTP)

CTS = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(TS1,TS2)

coarse_elements_to_refine = [3,4,5,8,9,10]
refined_elements = vcat(Mantis.FunctionSpaces.get_finer_elements.((CTS,), coarse_elements_to_refine)...)

refined_domains = Mantis.FunctionSpaces.HierarchicalActiveInfo([1:CTP_num_els;refined_elements], [0, CTP_num_els, CTP_num_els + length(refined_elements)])
hspace = Mantis.FunctionSpaces.get_hierarchical_space(spaces, [CTS], refined_domains)

x1, _ = Mantis.Quadrature.gauss_legendre(deg1+1)
x2, _ = Mantis.Quadrature.gauss_legendre(deg2+1)
xi = (x1, x2)

# Tests for coefficients and evaluation
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(hspace)

    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(hspace, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity

    # check Hierarchical B-spline evaluation
    h_eval, _ = Mantis.FunctionSpaces.evaluate(hspace, el, xi, 0)
    # Positivity of the basis
    @test minimum(h_eval[1][1]) >= 0.0
end

hspace.active_elements

# Test if projection in space is exact
nxi_per_dim = 3
nxi = nxi_per_dim^2
xi_per_dim = collect(range(0,1, nxi_per_dim))
xi = Matrix{Float64}(undef, nxi,2)

xi_eval = (xi_per_dim, xi_per_dim)

for (idx,x) ∈ enumerate(Iterators.product(xi_per_dim, xi_per_dim))
    xi[idx,:] = [x[1] x[2]]
end

xs = Matrix{Float64}(undef, Mantis.FunctionSpaces.get_num_elements(hspace)*nxi,2)
nx = size(xs)[1]

A = zeros(nx, Mantis.FunctionSpaces.get_num_basis(hspace))

for el ∈ 1:1:Mantis.FunctionSpaces.get_num_elements(hspace)
    level = Mantis.FunctionSpaces.get_active_level(hspace.active_elements, el)
    element_id = Mantis.FunctionSpaces.get_active_id(hspace.active_elements, el)

    max_ind_els = Mantis.FunctionSpaces._get_num_elements_per_space(hspace.spaces[level])
    ordered_index = Mantis.FunctionSpaces.linear_to_ordered_index(element_id, max_ind_els)

    borders_x = Mantis.Mesh.get_element(hspace.spaces[level].function_space_1.knot_vector.patch_1d, ordered_index[1])
    borders_y = Mantis.Mesh.get_element(hspace.spaces[level].function_space_2.knot_vector.patch_1d, ordered_index[2])

    x = [(borders_x[1] .+ xi[:,1] .* (borders_x[2] - borders_x[1])) (borders_y[1] .+ xi[:,2] .* (borders_y[2] - borders_y[1]))]

    idx = (el-1)*nxi+1:el*nxi
    xs[idx,:] = x

    local eval = Mantis.FunctionSpaces.evaluate(hspace, el, xi_eval, 0)

    A[idx, eval[2]] = eval[1][1][1]
end

coeffs = A \ xs
all(isapprox.(A * coeffs .- xs, 0.0, atol=1e-14))
