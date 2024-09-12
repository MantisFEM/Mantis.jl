import Mantis

import ReadVTK
using Printf
using Test
using LinearAlgebra

# Create the space

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

CTP = Mantis.FunctionSpaces.TensorProductSpace(CB1, CB2)
CTS, FTP = Mantis.FunctionSpaces.subdivide_bspline(CTP, (nsub1, nsub2))
spaces = [CTP, FTP]

CTP_num_els = Mantis.FunctionSpaces.get_num_elements(CTP)

coarse_elements_to_refine = [3,4,5,8,9,10,13,14,15]
refined_elements = vcat(Mantis.FunctionSpaces.get_finer_elements.((CTS,), coarse_elements_to_refine)...)

hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, [CTS], [Int[], refined_elements], true)

# Test if projection in space is exact
nxi_per_dim = max(deg1, deg2) + 1
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

hierarchical_geo = Mantis.Geometry.FEMGeometry(hspace, coeffs)

field_coeffs = Matrix{Float64}(LinearAlgebra.I,Mantis.FunctionSpaces.get_num_basis(hspace), Mantis.FunctionSpaces.get_num_basis(hspace))
tensor_field = Mantis.Fields.FEMField(hspace, field_coeffs)

# Generate the Plot
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "Geometry")

output_filename = "fem_geometry_tensor_hbsplines.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(hierarchical_geo, tensor_field; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)