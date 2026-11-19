module TensorProductBsplinesMaxwellEigenvalueTests

using Mantis

using Test
using DelimitedFiles

############################################################################################
#                                      Problem setup                                       #
############################################################################################
# Mesh
const starting_point = (0.0, 0.0)
const box_size = (Float64(pi), 1.0) # This size is so that the eigenvalues are unique.
const num_runs = 2 # Number of problem runs. Each solve 2≤i≤num_runs will have 2^i elements.
num_elements = [(2, 2) .^ i for i in 1:num_runs]

# B-spline parameters
p = (3, 3) # Polynomial degrees.
k = p .- 1 # Regularities. (Maximally smooth B-splines.)

# Quadrature rules
nq_assembly = p .+ 1
nq_error = nq_assembly .* 2
canonical_qrule_assembly, canonical_qrule_error = Quadrature.get_canonical_quadrature_rules(
    Quadrature.gauss_legendre, nq_assembly, nq_error
)

# Number of eigenvalues to compute
const num_eig = 5
# Scaling form maxwell eigenfunctions.
scale_factors = ntuple(2) do k
    return pi / (box_size[k] - starting_point[k])
end

############################################################################################
#                                       Run problem                                        #
############################################################################################
function run_problems(num_elements, p, k)
    eig_val_errors = [Vector{Float64}(undef, num_runs) for _ in 1:num_eig]
    eig_func_errors = [Vector{Float64}(undef, num_runs) for _ in 1:num_eig]
    num_dofs = zeros(Int, num_runs)
    for run_id in 1:num_runs
        curr_num_elements = num_elements[run_id]
        # Global quadrature rule for assembly
        qrule_assembly = Quadrature.StandardQuadrature(
            canonical_qrule_assembly, prod(curr_num_elements)
        )
        # create de Rham complex
        R_complex = Forms.create_tensor_product_bspline_de_rham_complex(
            starting_point, box_size, curr_num_elements, p, k
        )
        R0, R1 = R_complex[1], R_complex[2]
        num_dofs[run_id] = Forms.get_num_basis(R1)
        geom = Forms.get_geometry(R0)
        compt_eig_vals, compt_eig_funcs = Assemblers.solve_maxwell_eig(
            R0, R1, qrule_assembly, num_eig
        )
        exact_eig_vals, exact_eig_funcs = Assemblers.get_analytical_maxwell_eig(
            num_eig, geom, scale_factors
        )

        # Global quadrature rule for error computation
        qrule_error = Quadrature.StandardQuadrature(
            canonical_qrule_error, prod(curr_num_elements)
        )
        for eig_id in 1:num_eig
            eig_val_errors[eig_id][run_id] = compt_eig_vals[eig_id] - exact_eig_vals[eig_id]
            eig_func_errors[eig_id][run_id] = Analysis.L2_norm(
                compt_eig_funcs[eig_id] - exact_eig_funcs[eig_id], qrule_error
            )
        end
        @test all(isapprox.(compt_eig_vals, exact_eig_vals, atol=10.0^(1 - run_id)))
    end

    return eig_val_errors, eig_func_errors, num_dofs
end

eig_val_errors, eig_func_errors, num_dofs = run_problems(num_elements, p, k)

# eig_val_error_rates = [Vector{Float64}(undef, num_runs) for _ in 1:num_eig]
# eig_func_error_rates = [Vector{Float64}(undef, num_runs) for _ in 1:2]
# for eig_id in 1:num_eig
#     eig_val_error_rates[eig_id] =
#         log.(Ref(2), eig_val_errors[eig_id][1:(end - 1)] ./ eig_val_errors[eig_id][2:end])
#
#     @test isapprox(eig_val_error_rates[eig_id][end], 2 * minimum(p), atol=2e-1)
# end

end
