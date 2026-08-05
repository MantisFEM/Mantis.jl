module LChainTests

#=
The following tests are based on the work done in https://arxiv.org/abs/2502.19542.
=#

using Mantis
using Test

# Examples of figure 5.1
# Initial mesh.
starting_point = (0.0, 0.0)
box_size = (1.0, 1.0)
num_elements = (8, 7)
# B-spline parameters
p = (2, 2) # Polynomial degrees.
k = p .- 1 # Regularities.
B0 = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k)
# Hierarchical parameters
truncate = false
num_steps = 1 # Number of refinement steps.
num_sub = (2, 2) # Number of subdivisions per dimension per step.
## (a)
H = FunctionSpaces.HierarchicalFiniteElementSpace(B0, num_sub, truncate)
domains = FunctionSpaces.get_nested_domains(H)
ts = FunctionSpaces.get_two_scale_operators(H)
lin_num_basis = FunctionSpaces.get_lin_num_basis(B0)
input_i = (3, 4)
input_j = (8, 6)
βᵢ, βⱼ = lin_num_basis[input_i...], lin_num_basis[input_j...]
marked_elements = [mapreduce(β -> FunctionSpaces.get_support(B0, β), vcat, (βᵢ, βⱼ))]
Ωₗₗ = FunctionSpaces.get_level_domain(H, 2)
FunctionSpaces.refine_domains!(domains, ts, marked_elements[1], 1)
Blk = FunctionSpaces.get_Blk(H, 1)
@test length(Ωₗₗ) == 72
@test Blk == [31, 32, 33, 58, 59, 60]
@test isempty(FunctionSpaces.initiate_pairs(H, 1, Blk, marked_elements[1]))
## (b)
H = FunctionSpaces.HierarchicalFiniteElementSpace(B0, num_sub, truncate)
domains = FunctionSpaces.get_nested_domains(H)
ts = FunctionSpaces.get_two_scale_operators(H)
lin_num_basis = FunctionSpaces.get_lin_num_basis(B0)
input_i = (4, 4)
input_j = (6, 6)
input_t1 = (4, 5)
input_t2 = (5, 4)
βᵢ, βⱼ = lin_num_basis[input_i...], lin_num_basis[input_j...]
βₜ₁, βₜ₂ = lin_num_basis[input_t1...], lin_num_basis[input_t2...]
marked_elements = [
    mapreduce(β -> FunctionSpaces.get_support(B0, β), union, (βᵢ, βⱼ, βₜ₁, βₜ₂))
]
Ωₗₗ = FunctionSpaces.get_level_domain(H, 2)
FunctionSpaces.refine_domains!(domains, ts, marked_elements[1], 1)
Blk = FunctionSpaces.get_Blk(H, 1)
@test length(Ωₗₗ) == 84
@test Blk == [34, 35, 44, 45, 56]
initiate_pairs = FunctionSpaces.initiate_pairs(H, 1, Blk, marked_elements[1])
@test initiate_pairs == [
    (34, 44),
    (34, 35),
    (34, 45),
    (34, 56),
    (35, 44),
    (35, 45),
    (35, 56),
    (44, 45),
    (44, 56),
    (45, 56),
]
for pair in initiate_pairs
    @test FunctionSpaces.has_minimal_intersection(H, 1, pair)
    if 56 in pair
        @test !FunctionSpaces.has_shortest_chain(H, 1, Blk, pair)
    else
        @test FunctionSpaces.has_shortest_chain(H, 1, Blk, pair)
    end
end
## (c)
H = FunctionSpaces.HierarchicalFiniteElementSpace(B0, num_sub, truncate)
domains = FunctionSpaces.get_nested_domains(H)
ts = FunctionSpaces.get_two_scale_operators(H)
lin_num_basis = FunctionSpaces.get_lin_num_basis(B0)
input_i = (4, 4)
input_j = (6, 6)
input_t = (7, 4)
βᵢ, βⱼ = lin_num_basis[input_i...], lin_num_basis[input_j...]
βₜ = lin_num_basis[input_t...]
marked_elements = [mapreduce(β -> FunctionSpaces.get_support(B0, β), union, (βᵢ, βⱼ, βₜ))]
Ωₗₗ = FunctionSpaces.get_level_domain(H, 2)
FunctionSpaces.refine_domains!(domains, ts, marked_elements[1], 1)
Blk = FunctionSpaces.get_Blk(H, 1)
@test Blk == [34, 35, 36, 37, 46, 56]
initiate_pairs = FunctionSpaces.initiate_pairs(H, 1, Blk, marked_elements[1])
@test initiate_pairs == [(34, 56), (34, 37), (37, 56)]
for pair in initiate_pairs
    @test FunctionSpaces.has_minimal_intersection(H, 1, pair)
    @test FunctionSpaces.has_shortest_chain(H, 1, Blk, pair)
end
# Example of figure 7.1
# Initial mesh.
starting_point = (0.0, 0.0)
box_size = (1.0, 1.0)
num_elements = (10, 10)
# B-spline parameters
p = (3, 3) # Polynomial degrees.
k = p .- 1 # Regularities.
B0 = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k)
# Hierarchical parameters
truncate = false
num_steps = 1 # Number of refinement steps.
num_sub = (2, 2) # Number of subdivisions per dimension per step.
H = FunctionSpaces.HierarchicalFiniteElementSpace(B0, num_sub, truncate)
domains = FunctionSpaces.get_nested_domains(H)
ts = FunctionSpaces.get_two_scale_operators(H)
lin_num_basis = FunctionSpaces.get_lin_num_basis(B0)
input_t1 = (9, 5)
input_t2 = (7, 7)
input_t3 = (5, 9)
t1 = lin_num_basis[input_t1...]
t2 = lin_num_basis[input_t2...]
t3 = lin_num_basis[input_t3...]
marked_elements = [
    mapreduce(β -> FunctionSpaces.get_support(B0, β), union, (t1, t2, t3)), Int[]
]
FunctionSpaces.refine_domains!(domains, ts, marked_elements[1], 1)
Blk = FunctionSpaces.get_Blk(H, 1)
@test Blk == [61, 85, 109]
FunctionSpaces.update_domains_with_lchains!(H, marked_elements)
Blk = FunctionSpaces.get_Blk(H, 1)
@test Blk == [59, 60, 61, 72, 83, 84, 85, 96, 109]

end
