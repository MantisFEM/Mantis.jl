############################################################################################
#                                          Setup                                           #
############################################################################################

using Mantis
using BenchmarkTools
using Profile, PProf
include("Helpers.jl")

VERBOSE = true
EXPORT = false

# Function spaces
const starting_point = (0.0, 0.0)
const box_size = (1.0, 1.0)
const num_elements = (20, 20)
const p = (3, 3)
const k = (2, 2)

# Initial Spaces
B = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k)
nsubs = (2, 2)
nlevels = 5
CTP = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k)
TTS, FTP = Mantis.FunctionSpaces.build_two_scale_operator(CTP, nsubs)
spaces = [CTP, FTP]
operators = [TTS]
for level in 3:nlevels
    new_operator, new_space = Mantis.FunctionSpaces.build_two_scale_operator(
        spaces[level - 1], nsubs
    )
    push!(spaces, new_space)
    push!(operators, new_operator)
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

for (i, j) in Iterators.product(1:160, 141:160)
    push!(marked_elements_per_level[4], lin_4[i, j])
end

for (i, j) in Iterators.product(1:320, 301:320)
    push!(marked_elements_per_level[5], lin_5[i, j])
end


active_info = Hierarchy.ActiveInfo(marked_elements_per_level)
space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(
    spaces, operators, active_info, nsubs, true
)
@show FunctionSpaces.get_num_basis(space)
withref() = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(
    spaces, operators, active_info, nsubs, true
)
withref_bench = @benchmark withref()
open("withref_old.txt", "w") do io
    return show(io, MIME"text/plain"(), withref_bench)
end
# Profile.clear()
# @profile begin
#     for _ in 1:100
#         withref()
#     end
# end
# pprof(; web=false, out="withref.pb.gz")
