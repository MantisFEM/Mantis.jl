# # Adaptive refinement

# Uniform mesh refinement is wasteful when the solution is only hard to resolve in a small
# part of the domain. *Adaptive* refinement instead concentrates degrees of freedom where they
# are needed, guided by a local error indicator. `Mantis` supports this through
# **hierarchical B-splines**: a nested sequence of B-spline levels from which an active subset
# of basis functions is selected, refined locally, and kept compatible across the de Rham
# complex.

# In this example we solve the same [Maxwell eigenvalue problem](@ref) as before, but on an
# adaptively-refined hierarchical mesh.

# ## The adaptive loop

# Each adaptive step performs the classic cycle
# ```
# solve  →  estimate  →  mark  →  refine
# ```
# i.e. solve the discrete problem, estimate the error per element (here via
# [`Analysis.compute_error_per_element`](@ref Mantis.Analysis.compute_error_per_element)), mark
# the worst elements with a Dörfler (bulk-chasing) criterion, and refine them. `Mantis`
# bundles this whole loop for the Maxwell problem into the adaptive method of
# [`Assemblers.solve_maxwell_eig`](@ref Mantis.Assemblers.solve_maxwell_eig). Two ingredients
# make the refinement well-behaved:
#
# * a **Dörfler parameter** ``\theta \in (0, 1)`` controlling how aggressively to refine
#   (smaller = more localised), and
# * **L-chains**, which add a few extra elements to keep the hierarchical mesh *admissible*
#   (graded), so that the refined spaces still form a valid de Rham complex.
#
# The hierarchy bookkeeping itself lives in the [Hierarchy](@ref) and
# [FunctionSpaces](@ref) modules.

# ## Implementation

using Mantis
using Logging  # only used to keep the output of the adaptive loop tidy below

starting_point = (0.0, 0.0)
box_size       = (Float64(pi), Float64(pi))
num_elements   = (8, 8)    # initial (coarse) mesh

p = (2, 2)                 # B-spline degree
k = p .- 1                 # maximally-smooth B-splines

# Hierarchical-space parameters.
num_subdivisions = (2, 2)  # each refined element is split 2×2
truncate         = true    # use truncated hierarchical B-splines (THB-splines)
simplified       = false

# Adaptive-loop parameters.
num_steps         = 3      # number of refine-and-solve steps
dorfler_parameter = 0.2    # Dörfler bulk parameter θ
use_Lchains       = true   # keep the hierarchy admissible
eigenfunction     = 1      # which eigenfunction drives the error estimate
num_eig           = 5      # number of eigenvalues to track

# Quadrature for assembly and (finer) for error estimation.
nq_assembly = p .+ 1
nq_error    = nq_assembly .* 2
qrule_assembly, qrule_error = Quadrature.get_canonical_quadrature_rules(
    Quadrature.gauss_legendre, nq_assembly, nq_error
)
dΩₐ = Quadrature.StandardQuadrature(qrule_assembly, prod(num_elements))
dΩₑ = Quadrature.StandardQuadrature(qrule_error, prod(num_elements))

# Scale factors for the analytical Maxwell eigenfunctions on this domain.
scale_factors = ntuple(2) do d
    return pi / (box_size[d] - starting_point[d])
end

# Build the initial hierarchical de Rham complex. (As above, the `NullLogger` only suppresses
# an informational warning emitted while assembling the vector-valued 1-form space.)
complex = with_logger(NullLogger()) do
    return Forms.create_hierarchical_de_rham_complex(
        starting_point, box_size, num_elements, p, k, num_subdivisions, truncate, simplified
    )
end

num_elements_initial = Geometry.get_num_elements(Forms.get_geometry(complex[1]))

# Run the adaptive loop. The solver refines the complex `num_steps` times internally and
# returns the eigenvalues/eigenfunctions on the final, adapted mesh. The eigenfunctions live
# on that refined geometry, so we also read off how much the mesh grew and recompute the
# analytical eigenvalues there to measure the error. (We wrap everything in a `NullLogger`
# only to silence the informational compatibility warnings the internal routines emit; this
# has no effect on the result.)
computed_eigenvalues, num_elements_final, exact_eigenvalues = with_logger(NullLogger()) do
    eigenvalues, eigenfunctions = Assemblers.solve_maxwell_eig(
        complex,
        dΩₐ,
        num_steps,
        dorfler_parameter,
        dΩₑ,
        use_Lchains,
        eigenfunction,
        num_eig,
        scale_factors,
    )
    final_geometry = Forms.get_geometry(eigenfunctions...)
    exact_eigenvalues, _ = Assemblers.get_analytical_maxwell_eig(
        num_eig, final_geometry, scale_factors
    )
    return eigenvalues, Geometry.get_num_elements(final_geometry), exact_eigenvalues
end

println("Active elements: $num_elements_initial (initial) → $num_elements_final (after ",
        "$num_steps adaptive steps)")
for i in 1:num_eig
    error = abs(computed_eigenvalues[i] - exact_eigenvalues[i])
    println("  mode $i: ω² ≈ $(round(computed_eigenvalues[i]; digits=4)) ",
            "(error $(round(error; sigdigits=2)))")
end

# Only the elements flagged by the error estimator are refined, so the active-element count
# grows much more slowly than it would under uniform refinement, while the eigenvalues still
# converge towards their exact values. Swapping `use_Lchains` or `dorfler_parameter` changes
# how the refinement is distributed; experimenting with them is a good way to build intuition
# for adaptive methods.
