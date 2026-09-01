############################################################################################
#                                          Setup                                           #
############################################################################################

using Mantis
using Mantis.Hierarchical
using BenchmarkTools
using Profile, PProf
using JET, InteractiveUtils
include("Helpers.jl")

VERBOSE = true
EXPORT = false
PLOT = false

# Function spaces
const starting_point = (0.0, 0.0)
const box_size = (1.0, 1.0)
const num_elements = (20, 20)
const p = (3, 3)
const k = (2, 2)

ref_geo_h2(g) = Geometry.refinement_uniform(g, 2)
ref_space_h2(s) = FunctionSpaces.refinement_uniform(s, 2)
scal_space_h2 = FunctionSpaces.scaling_matrix_uniform
const SelStd = FunctionSpaces.SelectionStandard
const SelSimp = FunctionSpaces.SelectionSimple
const THB = FunctionSpaces.THB

# Initial Spaces
nlevels = 4
B1 = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k)
B2 = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements .* 2, p, k)
B3 = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements .* 4, p, k)
B4 = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements .* 8, p, k)
B5 = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements .* 16, p, k)
scal_1 = MatrixScaling(B1, B2, (scal_space_h2, scal_space_h2))
scal_2 = MatrixScaling(B2, B3, (scal_space_h2, scal_space_h2))
scal_3 = MatrixScaling(B3, B4, (scal_space_h2, scal_space_h2))
scal_4 = MatrixScaling(B4, B5, (scal_space_h2, scal_space_h2))
spaces = (B1, B2, B3, B4, B5)
scalings = (scal_1, scal_2, scal_3, scal_4)

# Hierarchical Geometry
geo_l1 = FunctionSpaces.get_geometry(B1)
geo_scal_1 = Geometry.scaling_uniform(geo_l1, 2)
geo_scal_2 = Geometry.scaling_uniform(get_child(geo_scal_1), 2)
geo_scal_3 = Geometry.scaling_uniform(get_child(geo_scal_2), 2)
geo_scal_4 = Geometry.scaling_uniform(get_child(geo_scal_3), 2)

lin_1 = LinearIndices((20, 20))
lin_2 = LinearIndices((40, 40))
lin_3 = LinearIndices((80, 80))
lin_4 = LinearIndices((160, 160))
lin_5 = LinearIndices((320, 320))
marked_elements_per_level = [Int[], Int[], Int[], Int[], Int[]]

for (i, j) in Iterators.product(1:20, 1:10)
    push!(marked_elements_per_level[1], lin_1[i, j])
end

for (i, j) in Iterators.product(1:40, 21:30)
    push!(marked_elements_per_level[2], lin_2[i, j])
end

for (i, j) in Iterators.product(1:80, 61:70)
    push!(marked_elements_per_level[3], lin_3[i, j])
end

for (i, j) in Iterators.product(1:160, 141:150)
    push!(marked_elements_per_level[4], lin_4[i, j])
end

for (i, j) in Iterators.product(1:320, 301:320)
    push!(marked_elements_per_level[5], lin_5[i, j])
end

active_info = ActiveInfo(marked_elements_per_level)
hgeo = Geometry.HierarchicalGeometry(
    NestedHierarchy(active_info, geo_scal_1, geo_scal_2, geo_scal_3, geo_scal_4)
)
space = FunctionSpaces.HierarchicalSpace(hgeo, hgeo, scalings, SelStd, THB)
geometry = FunctionSpaces.get_geometry(space)

canonical_qrule = Quadrature.tensor_product_rule(p .+ 1, Quadrature.gauss_legendre)
dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geometry))
f = Forms.AnalyticalFormField(0, x -> [ones(size(x, 1))], geometry, "f")
f_space = FormSpace(0, space, "fₕ")
fₕ = Assemblers.solve_L2_projection(f_space, f, dΩ)
@show Analysis.L2_norm(f - fₕ, dΩ)

solve() = Assemblers.solve_L2_projection(f_space, f, dΩ)
bench = @benchmark solve()
display(bench)
exit()
# open("solve_new.txt", "w") do io
#     return show(io, MIME"text/plain"(), bench)
# end

# Profile.clear()
# @profile begin
#     for _ in 1:10
#         solve()
#     end
# end
# pprof(; from_c=false, out="solve_new.pb.gz")
# exit()

# Profile.Allocs.clear()
# Profile.Allocs.@profile solve()
# PProf.Allocs.pprof(; from_c=false, out="solve_new_allocs.pb.gz")
