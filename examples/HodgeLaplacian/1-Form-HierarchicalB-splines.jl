module HierarchicalBsplines

using Mantis

# Refer to the following file for method and variable definitions
include("../HelperFunctions.jl")

############################################################################################
#                                      Problem setup                                       #
############################################################################################
# Mesh
starting_point = (0.0, 0.0)
box_size = (1.0, 1.0)
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

# Quadrature rules
nq_assembly = p .+ 1
nq_error = nq_assembly .* 2
∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(
    Quadrature.gauss_legendre, nq_assembly, nq_error
)

forcing_function = sinusoidal_data

verbose = true # Set to true for problem information.
export_vtk = false # Set to true to export the computed solutions.

############################################################################################
#                                       Run problem                                        #
############################################################################################
# Hierarchical de Rham complex
ℌ = Forms.create_hierarchical_de_rham_complex(
    starting_point, box_size, num_elements, p, k, num_sub, truncate, simplified
)

# Solve problem
uₕ, δuₕ = Assemblers.solve_one_form_hodge_laplacian(
    ℌ, forcing_function, num_steps, θ, Lchains, ∫ₐ, ∫ₑ; verbose
)

############################################################################################
#                                      Solution data                                       #
############################################################################################
if export_vtk
    println("Exporting computed solutions to VTK...")
    geometry = Forms.get_geometry(uₕ)
    num_elements = Geometry.get_num_elements(geometry)
    u, δu = forcing_function(1, geometry)

    compt_file_name = "1-Form-HodgeLaplacian-HBsplines-computed-p=$(p)-k=$(k)-nels=$(num_elements)"
    exact_file_name = "1-Form-HodgeLaplacian-HBsplines-exact-p=$(p)-k=$(k)-nels=$(num_elements)"

    Plot.export_form_fields_to_vtk((uₕ, δuₕ), compt_file_name)
    Plot.export_form_fields_to_vtk((u, δu), exact_file_name)
end

end
