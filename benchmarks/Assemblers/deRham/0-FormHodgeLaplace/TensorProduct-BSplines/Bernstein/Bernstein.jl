module OneFormTensorProductHodgeLaplaceBenchmarks

using Mantis

# Refer to the following file for method and variable definitions
include(joinpath(pwd(), "BenchmarkHelpers.jl"))
include(joinpath(mantis_dir, "examples", "HelperFunctions.jl"))

############################################################################################
#                                      Problem setup                                       #
############################################################################################
function problem_setup(manifold_dim::Int, num_els::Int, p::Int, k::Int)
    # Mesh
    starting_point = ntuple(_ -> 0.0, manifold_dim)
    box_size = ntuple(_ -> 1.0, manifold_dim)
    num_els = ntuple(_ -> num_els, manifold_dim)

    # B-spline parameters
    p_tuple = ntuple(_ -> p, manifold_dim)
    k_tuple = ntuple(_ -> k, manifold_dim)

    # Quadrature rules
    nq_assembly = p_tuple .+ 1
    nq_error = nq_assembly .* 2
    ∫ₐ, _ = Mantis.Quadrature.get_canonical_quadrature_rules(
        Mantis.Quadrature.gauss_legendre, nq_assembly, nq_error
    )
    dΩₐ = Quadrature.StandardQuadrature(∫ₐ, prod(num_els))
    ℜ⁰, ℜ¹, _ = Mantis.Forms.create_tensor_product_bspline_de_rham_complex(
        starting_point, box_size, num_els, p_tuple, k_tuple
    )

    _, _, fₑ = sinusoidal_data(0, Mantis.Forms.get_geometry(ℜ⁰))

    return ℜ⁰, ℜ¹, fₑ, dΩₐ
end

function run_problem(manifold_dim::Int, num_els::Int, p::Int, k::Int)
    ℜ⁰, ℜ¹, fₑ, dΩₐ = problem_setup(manifold_dim, num_els, p, k)
    weak_form_inputs = Assemblers.WeakFormInputs((ℜ⁰,), (fₑ,))
    lhs_expressions, rhs_expressions = Assemblers.zero_form_hodge_laplacian(
        weak_form_inputs, dΩₐ
    )
    weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, weak_form_inputs)
    num_dofs = Assemblers.get_trial_size(weak_form)

    return num_dofs,
    @benchmarkable Assemblers.assemble($weak_form) samples = 10 evals = 1 seconds = Inf
end

############################################################################################
#                                       Run problems                                        #
############################################################################################

group = BenchmarkGroup()
manifold_dim = [2, 3]
num_elements = [2 .^ (2:4), 2 .^ (2:3)] # For each manifold_dim
p = 3
k = p - 1
for i in eachindex(manifold_dim)
    sub_group = BenchmarkGroup()
    for num_els in num_elements[i]
        num_dofs, benchmarkable = run_problem(manifold_dim[i], num_els, p, k)
        name = "num_dofs=$(num_dofs)-p=$(p)-k=$(k)"
        sub_group[name] = benchmarkable
    end

    group["$(manifold_dim[i])D"] = sub_group
end

end
