"""
	MaxwellEigenvalue

This example was used in the sub-section `7.2 Maxwell eigenvalue problem` of [Cabanas2025](@cite).
"""
module MaxwellEigenvalue

using Mantis

import CSV, DataFrames

# Refer to the following file for method and variable definitions
include("../HelperFunctions.jl")

############################################################################################
#                                      Problem setup                                       #
############################################################################################
# Mesh
starting_point = (0.0, 0.0)
box_size = (Float64(π), Float64(π))
num_elements = (15, 15) # Ininital mesh size.

# B-spline parameters
p = (4, 4) # Polynomial degrees.
k = p .- 1 # Regularities. (Maximally smooth B-splines.)

# Hierarchical parameters.
truncate = false # true = THB, false = HB
simplified = false
num_steps = 3 # Number of refinement steps.
num_sub = (2, 2) # Number of subdivisions per dimension per step.

# Create the Hierarchical space for each figure (a and b). Still tensor-product because we
# have not refined them.
𝔅_a = Forms.create_hierarchical_de_rham_complex(
    starting_point, box_size, num_elements, p, k, num_sub, truncate, simplified
)
𝔅_b = Forms.create_hierarchical_de_rham_complex(
    starting_point, box_size, num_elements, p, k, num_sub, truncate, simplified
)

# Define the refinement domains
geometry = Geometry.get_base_geometry(Forms.get_geometry(𝔅_a[1]))
marked_elements = union(
    get_elements_in_box(geometry, (6, 3), (10, 13)),
    get_elements_in_box(geometry, (3, 6), (13, 10)),
)

## Figure 8. a)
center_element = ceil(Int, prod(num_elements) / 2)
FunctionSpaces.update_space!(
    Forms.get_fe_space(𝔅_a[1]), [setdiff(marked_elements, center_element), Int[]]
)
ℌ_a = Forms.update_hierarchical_de_rham_complex(𝔅_a, Forms.get_fe_space(𝔅_a[1]))

## Figure 8. b)
FunctionSpaces.update_space!(Forms.get_fe_space(𝔅_b[1]), [marked_elements, Int[]])
ℌ_b = Forms.update_hierarchical_de_rham_complex(𝔅_b, Forms.get_fe_space(𝔅_b[1]))

# Number of eigenvalues to compute
num_eig = 50
# Scaling form maxwell eigenfunctions.
scale_factors = ntuple(2) do k
    return pi / (box_size[k] - starting_point[k])
end

verbose = true # Set to true for problem information.
export_vtk = false # Set to true to export the computed eigenfunctions.
export_csv = true # Set to true to export the computed eigenvalues.

############################################################################################
#                                       Run problem                                        #
############################################################################################

# Quadrature rules
nq_assembly = 2 .* (p .+ 1)
nq_error = nq_assembly .* 2
∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(
    Quadrature.gauss_legendre, nq_assembly, nq_error
)
dΩ_a = Quadrature.StandardQuadrature(∫ₐ, Forms.get_num_elements(ℌ_a[1]))
dΩ_b = Quadrature.StandardQuadrature(∫ₐ, Forms.get_num_elements(ℌ_b[1]))

# Solve problem (a)
ωₕ²_a, uₕ_a = Assemblers.solve_maxwell_eig(ℌ_a[1], ℌ_a[2], dΩ_a, num_eig; verbose=verbose)
# Solve problem (b)
ωₕ²_b, uₕ_b = Assemblers.solve_maxwell_eig(ℌ_b[1], ℌ_b[2], dΩ_b, num_eig; verbose=verbose)

############################################################################################
#                                      Solution data                                       #
############################################################################################

function print_data(ωₕ², uₕ; offset=0)
    geometry = Forms.get_geometry(uₕ[1])
    # Exact eigenvalues
    ω² = Assemblers.get_analytical_maxwell_eig(num_eig, geometry, scale_factors)[1]

    print("Printing first $(num_eig-offset) exact and computed eigenvalues...")
    iszero(offset) ? print("\n") : println(" (After removing $offset eigenvalues.)")
    println("i    ω²  ωₕ²  (ωₕ²[i] - ω²[i])^2")
    println("--   --  --   ------------------")
    for i in 1:(num_eig - offset)
        @printf "%02.f" i
        @printf "   %02i" ω²[i]
        @printf "   %02i" ωₕ²[i + offset]
        @printf "   %e\n" (ωₕ²[i + offset] - ω²[i])
    end

    return nothing
end

function export_vtk_data(uₕ, figure=String)
    println("Exporting computed eigenfunctions to VTK...")
    hier_num_elements = Geometry.get_num_elements(Forms.get_geometry(uₕ[1]))
    file_base_name = "Figure8$(figure)-computed-p=$(p)-k=$(k)-nels=$(hier_num_elements)"
    labels = Vector{String}(undef, num_eig)
    for i in 1:num_eig
        labels[i] = uₕ[i].label
    end

    Plot.export_form_fields_to_vtk(uₕ, labels, file_base_name)

    return nothing
end

function export_csv_data(ωₕ², uₕ, figure::String; offset=0, last_eig=1)
    eig_ids = 1:(last_eig - offset)
    filename = "maxwell-eigenvalue-figure8$(figure)"
    filename = joinpath("examples", "L-chains", filename)
    geometry = Forms.get_geometry(uₕ[1])
    ω² = Assemblers.get_analytical_maxwell_eig(num_eig, geometry, scale_factors)[1]
    computed_vals = ωₕ²[(1 + offset):(last_eig)]
    error = abs.(ω²[1:(last_eig - offset)] .- computed_vals)
    exact_df = DataFrames.DataFrame(; id=eig_ids, eigenvalue=ω²[1:(last_eig - offset)])
    approximate_df = DataFrames.DataFrame(;
        id=eig_ids, eigenvalue=computed_vals, error=error
    )
    CSV.write(filename * "-exact.csv", exact_df)
    CSV.write(filename * "-approximate.csv", approximate_df)

    return nothing
end

if verbose
    println("\nData from Figure 8. a)\n")
    offset = 4 # number of spurious harmonics in Figure 8. a)
    print_data(ωₕ²_a, uₕ_a; offset=offset)
    println("\nData from Figure 8. b)\n")
    print_data(ωₕ²_b, uₕ_b)
end

if export_vtk
    println("\nExporting vtk data from Figure 8. a)\n")
    export_vtk_data(uₕ_a, "a")
    println("\nExporting vtk data from Figure 8. b)\n")
    export_vtk_data(uₕ_b, "b")
end

if export_csv
    println("\nExporting vtk data from Figure 8. a)\n")
    offset = 4 # number of spurious harmonics in Figure 8. a)
    export_csv_data(ωₕ²_a, uₕ_a, "a"; offset=offset, last_eig=num_eig)
    println("\nExporting vtk data from Figure 8. b)\n")
    export_csv_data(ωₕ²_b, uₕ_b, "b"; offset=0, last_eig=num_eig - offset)
end

end
