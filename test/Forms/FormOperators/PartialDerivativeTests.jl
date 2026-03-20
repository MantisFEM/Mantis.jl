module PartialDerivativeTests

using Mantis

using Test
using LinearAlgebra

############################################################################################
#                                          Setup                                           #
############################################################################################

verbose = false
export_vtk = false
debug = false

# Geometry
starting_point = (0.0, 0.0)
box_size = (1.0, 1.0)
num_elements = (4, 5)

# B-spline parameters
p = (4, 4) # Polynomial degrees
k = p .- 1 # Regularity

# Quadrature rules
nq_assembly = p .+ 1
nq_error = nq_assembly .* 2
∫ₐ, ∫ₑ = Quadrature.get_canonical_quadrature_rules(
    Quadrature.gauss_legendre, nq_assembly, nq_error
)
dΩₐ = Quadrature.StandardQuadrature(∫ₐ, prod(num_elements))
dΩₑ = Quadrature.StandardQuadrature(∫ₑ, prod(num_elements))

############################################################################################
#                                         Spaces                                           #
############################################################################################

function forcing(x::Matrix{Float64}, order)
    result = [ones(size(x, 1))]
    for dim in 1:2
        if order[dim] == 0
            @. result[1] *= x[:, dim]^2 * (1.0 - x[:, dim])^2
        elseif order[dim] == 1
            @. result[1] *= 4 * x[:, dim]^3 - 6 * x[:, dim]^2 + 2 * x[:, dim]
        elseif order[dim] == 2
            @. result[1] *= 12 * x[:, dim]^2 - 12 * x[:, dim] + 2
        else
            return [zeros(size(x, 1))]
        end
    end

    return result
end

orders = ((1, 0), (0, 1), (1, 1), (2, 0), (0, 2), (2, 1), (1, 2), (2, 2))
forcings = Dict(order => (x -> forcing(x, order)) for order in orders)
B = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k)
G = FunctionSpaces.get_geometry(B)
f = Forms.AnalyticalFormField(0, x -> forcing(x, (0, 0)), G, "f")
∂f = Dict(
    order => Forms.AnalyticalFormField(0, forcings[order], G, "∂f") for order in orders
)
fₕ = Forms.FormSpace(0, B, "fₕ")

############################################################################################
#                                        Helper                                            #
############################################################################################
function solve_problem(fₕ, f, dΩ, order)
    wfi = Assemblers.WeakFormInputs(fₕ, f)
    wf = weak_form(wfi, dΩ, order)
    num_basis_univariate = (p[1] + num_elements[1], p[2] + num_elements[2])
    lin_basis_indices = LinearIndices(num_basis_univariate)
    bc_basis_indices = Int[]
    if order[2] > 0
        for basis_id in 1:num_basis_univariate[1]
            append!(bc_basis_indices, lin_basis_indices[basis_id, 1])
        end
    end

    if order[1] > 0
        for basis_id in 1:num_basis_univariate[2]
            append!(bc_basis_indices, lin_basis_indices[1, basis_id])
        end
    end

    bc = Dict(i => 0.0 for i in bc_basis_indices)
    A, b = Assemblers.assemble(wf, bc)
    debug ? display(cond(Matrix(A))) : nothing
    debug ? display(size(A, 1) - rank(A)) : nothing
    sol = vec(A \ b)
    fₕ = Forms.build_form_field(fₕ, sol; label="u")

    return ∂(fₕ, order)
end

function weak_form(inputs, dΩ, order)
    vᵏ = Assemblers.get_test_form(inputs)
    uᵏ = Assemblers.get_trial_form(inputs)
    fᵏ = Assemblers.get_forcing(inputs)
    fi = findfirst(i -> i > 0, order)
    u_order = ntuple(i -> i == fi ? (order[i] - 1) : order[i], 2)
    v_order = ntuple(i -> i == fi ? 1 : 0, 2)
    A = -∫(∂(vᵏ, v_order) ∧ ★(∂(uᵏ, u_order)), dΩ)
    b = ∫(vᵏ ∧ ★(fᵏ), dΩ)
    lhs_expression = ((A,),)
    rhs_expression = ((b,),)

    return Assemblers.WeakForm(lhs_expression, rhs_expression, inputs)
end

function build_partial_form_field(fₕ, order)
    form_space = Forms.get_form(fₕ)
    partial_form_space = ∂(form_space, order)
    ∂fₕ = Forms.build_form_field(
        partial_form_space, Forms.get_coefficients(fₕ); label="∂fₕ"
    )

    return ∂fₕ
end

############################################################################################
#                                          Tests                                           #
############################################################################################

one_form_error = Forms.FormSpace(1, FunctionSpaces.DirectSumSpace((B, B)), "u")
@test_throws MethodError ∂(one_form_error, (1, 1))
@test_throws ArgumentError ∂(fₕ, (-1, 0))
@test_throws ArgumentError ∂(fₕ, (0, -1))
@test_throws MethodError ∂(fₕ, (1,))
@test_throws MethodError ∂(fₕ, (0, 1, 0))
# Non-cartesian Geometry
mapping = Geometry.Mapping(Val(1), Val(1), x -> x[1], x -> zero(x))
mapped_B = FunctionSpaces.BSplineSpace(
    Geometry.create_cartesian_box((0.0,), (1.0,), (4,)), mapping, 1, 0
)
@test_throws ArgumentError ∂(Forms.FormSpace(0, mapped_B, "ω"), (1,))

# Manufactured solutions
L2_fₕ = Assemblers.solve_L2_projection(fₕ, f, dΩₐ)
verbose ? println("Running manufactured solution tests...") : nothing
for order in orders
    verbose ? println("\tOrder=$(order)...") : nothing
    ∂fₕ = build_partial_form_field(L2_fₕ, order)
    error_l2 = Analysis.L2_norm(∂fₕ - ∂f[order], dΩₑ)
    @test isapprox(error_l2, 0.0, atol=1e-11)
    verbose ? println("\t\tL2 projection error=$(error_l2).") : nothing
    if sum(order) <= 2
        ∂fₕ = solve_problem(fₕ, ∂f[order], dΩₐ, order)
        error_∂ = Analysis.L2_norm(∂fₕ - ∂f[order], dΩₑ)
        @test isapprox(error_∂, 0.0, atol=1e-15)
        verbose ? println("\t\t∂ weak form error=$(error_∂).") : nothing
    end
end

if export_vtk
    name = "PartialDerivative-Test"
    labels = ("fₕ", "∂ₓ₁fₕ", "∂ₓ₂fₕ", "∂ₓ₁∂ₓ₂fₕ", "∂ₓ₁∂ₓ₁fₕ", "∂ₓ₂∂ₓ₂fₕ")
    Plot.export_form_fields_to_vtk(
        (L2_fₕ, ∂(fₕ, (0, 1)), ∂(fₕ, (1, 1)), ∂(fₕ, (2, 0)), ∂(fₕ, (0, 2))),
        labels,
        name,
    )
end

end
