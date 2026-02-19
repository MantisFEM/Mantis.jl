"""
	HodgeLaplacian

This example was used in the sub-section `7.1 Vector Laplace problem` of [Cabanas2025](@cite).
"""
module HodgeLaplacian

using Mantis

# Refer to the following file for method and variable definitions
include("../HelperFunctions.jl")

VERBOSE = true
EXPORT = true
# Example of figure 7.4
# Initial mesh.
starting_point = (0.0, 0.0)
box_size = (1.0, 1.0)
num_elements = (16, 16)
# B-spline parameters
p = (2, 2) # Polynomial degrees.
k = p .- 1 # Regularities.
B0 = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k)
# Hierarchical parameters
truncate = true
simplified = false
num_steps = 2 # Number of refinement steps.
num_sub = (2, 2) # Number of subdivisions per dimension per step.
θ = 0.25 # Dorfler parameter.
Lchains = false # Decide if Lchains are added to fix inexact refinements.
# Quadrature rules
nq_assembly = p .+ 1
nq_error = nq_assembly .* 2
∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(
    Quadrature.gauss_legendre, nq_assembly, nq_error
)
dΩₐ, dΩₑ = (
    Quadrature.StandardQuadrature(∫ₐ, prod(num_elements)),
    Quadrature.StandardQuadrature(∫ₑ, prod(num_elements)),
)
# Hierarchical de Rham complex
ℍ = Forms.create_hierarchical_de_rham_complex(
    starting_point, box_size, num_elements, p, k, num_sub, truncate, simplified
)
# Solve problem
δuₕ, uₕ = Assemblers.solve_one_form_hodge_laplacian(
    ℍ,
    circular_data,
    dΩₐ,
    num_steps,
    θ,
    dΩₑ,
    Lchains,
    "dirichlet";
    VERBOSE,
    EXPORT,
    Plot=Mantis.Plot,
)

end
