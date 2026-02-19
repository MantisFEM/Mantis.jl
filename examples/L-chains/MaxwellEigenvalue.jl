module HierarchicalBsplines

using Mantis

using DelimitedFiles

# Refer to the following file for method and variable definitions
include("../HelperFunctions.jl")

############################################################################################
#                                      Problem setup                                       #
############################################################################################
# Mesh
starting_point = (0.0, 0.0)
box_size = (fpi, fpi) #(π, π)
num_elements = (2, 2) .^ 3 # Ininital mesh size.

# B-spline parameters
p = (2, 2) # Polynomial degrees.
k = p .- 1 # Regularities. (Maximally smooth B-splines.)

# Hierarchical parameters.
truncate = true # true = THB, false = HB
simplified = false
num_steps = 3 # Number of refinement steps.
num_sub = (2, 2) # Number of subdivisions per dimension per step.
θ = 0.2 # Dorfler parameter.
Lchains = true # Decide if Lchains are added to fix inexact refinements.
eigenfunc = 1 # Eigenfunction to use for adaptive refinement.

# Quadrature rules
nq_assembly = p .+ 1
nq_error = nq_assembly .* 2
∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(
    Quadrature.gauss_legendre, nq_assembly, nq_error
)
dΩₐ = Quadrature.StandardQuadrature(∫ₐ, prod(num_elements))
dΩₑ = Quadrature.StandardQuadrature(∫ₑ, prod(num_elements))

# Number of eigenvalues to compute
num_eig = 10
# Scaling form maxwell eigenfunctions.
scale_factors = ntuple(2) do k
    return pi / (box_size[k] - starting_point[k])
end

verbose = true # Set to true for problem information.
export_csv = false # Set to true to export the computed eigenvalues.
export_vtk = false # Set to true to export the computed eigenfunctions.

############################################################################################
#                                       Run problem                                        #
############################################################################################
# Hierarchical de Rham complex
ℌ = Forms.create_hierarchical_de_rham_complex(
    starting_point, box_size, num_elements, p, k, num_sub, truncate, simplified
)
# Solve problem
ωₕ², uₕ = Assemblers.solve_maxwell_eig(
    ℌ, dΩₐ, num_steps, θ, dΩₑ, Lchains, eigenfunc, num_eig, scale_factors; verbose
)

############################################################################################
#                                      Solution data                                       #
############################################################################################
# Print eigenvalues

if verbose
    ⊞ = Forms.get_geometry(uₕ[1])
    # Exact eigenvalues
    ω² = Assemblers.get_analytical_maxwell_eig(num_eig, ⊞, scale_factors)[1]

    println("Printing first $(num_eig) exact and computed eigenvalues...")
    println("i    ω²[i]      ωₕ²[i]     (ωₕ²[i] - ω²[i])^2")
    println("--   --------   --------   --------")
    for i in 1:num_eig
        @printf "%02.f" i
        @printf "   %08.5f" ω²[i]
        @printf "   %08.5f" ωₕ²[i]
        @printf "   %08.5f\n" (ωₕ²[i] - ω²[i])^2
    end
end

if export_vtk
    println("Exporting computed eigenfunctions to VTK...")
    hier_num_elements = Geometry.get_num_elements(Forms.get_geometry(uₕ[1]))
    file_base_name = "MaxwellEigenvalueHBsplines-computed-p=$(p)-k=$(k)-nels=$(hier_num_elements)"
    labels = Vector{String}(undef, num_eig)
    for i in 1:num_eig
        labels[i] = uₕ[i].label
    end

    Plot.export_form_fields_to_vtk(uₕ, labels, file_base_name)
end
if export_csv
    eig_ids = 1:num_eig
    eigenvalue_offset = 0
    filename = "MaxwellEigenvaluesHBSplines"
    writedlm(filename * "-exact.csv", [eig_ids ω²])
    writedlm(
        filename * "-approximate.csv",
        [eig_ids ωₕ²[(1 + eigenvalue_offset):(num_eig + eigenvalue_offset)]],
    )
end

end
