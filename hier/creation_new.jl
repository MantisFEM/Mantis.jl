############################################################################################
#                                          Setup                                           #
############################################################################################

using Mantis
using Mantis.Hierarchical
using BenchmarkTools
using PProf, Profile
using InteractiveUtils, JET
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
B1 = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k)
scal_1 = FunctionSpaces.scaling_uniform(B1, (2, 2))
scal_2 = FunctionSpaces.scaling_uniform(get_child(scal_1), (2, 2))
scal_3 = FunctionSpaces.scaling_uniform(get_child(scal_2), (2, 2))
scal_4 = FunctionSpaces.scaling_uniform(get_child(scal_3), (2, 2))
scalings = (scal_1, scal_2, scal_3, scal_4)

# Hierarchical Geometry
geo_scals = ntuple(4) do i
    return Geometry.scaling_uniform(
        FunctionSpaces.get_geometry(get_parent(scalings[i])),
        FunctionSpaces.get_geometry(get_child(scalings[i])),
        2
    )
end

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
hgeo = Geometry.HierarchicalGeometry(NestedHierarchy(active_info, geo_scals...))

space = FunctionSpaces.HierarchicalSpace(hgeo, hgeo, scalings, SelStd, THB)

withref() = FunctionSpaces.HierarchicalSpace(hgeo, hgeo, scalings, SelStd, THB)
withref_bench = @benchmark withref()
display(withref_bench)
# open("withref_new.txt", "w") do io
#     return show(io, MIME"text/plain"(), withref_bench)
# end

# geocreation() = Geometry.HierarchicalGeometry(
#     NestedHierarchy(
#         active_info, geo_scal_1, geo_scal_2, geo_scal_3, geo_scal_4
#     ),
# )
# basiscreation_simp() = FunctionSpaces.create_basis(hgeo, scalings, SelSimp)
# basiscreation_std() = FunctionSpaces.create_basis(hgeo, scalings, SelStd)
# display(@benchmark geocreation())
# display(@benchmark basiscreation_std())
# display(@benchmark basiscreation_simp())

# Profile.clear()
# @profile begin
#     for _ in 1:100
#         withref()
#     end
# end
# @profile withref()
# pprof(; from_c=false, out="withref_new.pb.gz")
#
exit()
Profile.Allocs.clear()
Profile.Allocs.@profile withref()
PProf.Allocs.pprof(; from_c=false, out="withref_new_allocs.pb.gz")
