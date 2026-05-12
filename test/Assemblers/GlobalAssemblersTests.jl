module GlobalAssemblersTests

using Mantis
using Test
using SparseArrays
const sp = SparseArrays

############################################################################################
#                                   Boundary Conditions                                    #
############################################################################################

lhs_rows = [1, 2, 3, 1, 2, 3, 1, 2, 3, 2]
lhs_cols = [1, 1, 1, 2, 2, 2, 3, 3, 3, 2] # lhs[2, 2] is duplicated
lhs_vals = fill(10.0, length(lhs_rows))
rhs_rows = [1, 2, 3, 2, 2] # rhs[2] is triplicated
rhs_cols = [1, 1, 1, 1, 1]
rhs_vals = fill(10.0, length(rhs_rows))
bc_id = 2
bc_value = 42.0
bc = Dict(bc_id => bc_value)
lhs_vec_truth = fill(10.0, (3, 3))
lhs_vec_truth[bc_id, :] .= 0.0
lhs_vec_truth[bc_id, bc_id] = 1.0
rhs_vec_truth = fill(10.0, 3)
rhs_vec_truth[bc_id] = bc_value
lhs_mat_truth = fill(10.0, (2, 2))
rhs_mat_truth = fill(10.0, (2, 1))
Assemblers.zero_rows!(lhs_vals, rhs_vals, lhs_rows, rhs_rows, bc)
@test lhs_vals == [10.0, 0.0, 10.0, 10.0, 0.0, 10.0, 10.0, 0.0, 10.0, 0.0]
@test rhs_vals == [10.0, 0.0, 10.0, 0.0, 0.0]
types = (Matrix{Float64}, sp.SparseMatrixCSC{Float64, Int})
for T in types
    lhs = Assemblers.build_array(T, lhs_rows, lhs_cols, lhs_vals, (3, 3))
    rhs = Assemblers.build_array(Vector{Float64}, rhs_rows, rhs_cols, rhs_vals, (3, 1))
    Assemblers.add_bc!(lhs, rhs, bc)
    @test lhs == lhs_vec_truth
    @test rhs == rhs_vec_truth
    rhs = Assemblers.build_array(Matrix{Float64}, rhs_rows, rhs_cols, rhs_vals, (3, 1))
    lhs_copy = copy(lhs)
    rhs_copy = copy(rhs)
    lhs, rhs = Assemblers.add_bc!(lhs, rhs, bc)
    @test lhs == lhs_mat_truth
    @test rhs == rhs_mat_truth
    lhs = copy(lhs_copy)
    rhs_copy = copy(rhs)
end

B = Forms.create_tensor_product_bspline_de_rham_complex(
    (0.0, 0.0), (1.0, 1.0), (1, 1), (1, 1), (0, 0)
)
dΩ = Quadrature.get_global_quadrature_rules(Quadrature.gauss_legendre, 1, (2, 2))[1]
function zero_rhs_weak_form(inputs, dΩ)
    B0, B1, B2 = Assemblers.get_trial_forms(inputs)
    lhs_expressions = (
        (∫(B0 ∧ ★(B0), dΩ), 0, 0), (0, ∫(B1 ∧ ★(B1), dΩ), 0), (0, 0, ∫(B2 ∧ ★(B2), dΩ))
    )
    rhs_expressions = ((0,), (0,), (0,))

    return Assemblers.WeakForm(lhs_expressions, rhs_expressions, inputs)
end

function with_rhs_weak_form(inputs, dΩ)
    B0, B1, B2 = Assemblers.get_trial_forms(inputs)
    f = Assemblers.get_forcing(inputs)
    lhs_expressions = (
        (∫(B0 ∧ ★(B0), dΩ), 0, 0), (0, ∫(B1 ∧ ★(B1), dΩ), 0), (0, 0, ∫(B2 ∧ ★(B2), dΩ))
    )
    rhs_expressions = ((0,), (∫(B1 ∧ ★(f), dΩ),), (0,))

    return Assemblers.WeakForm(lhs_expressions, rhs_expressions, inputs)
end

bc = Dict(2 => 42.0, 6 => 26.0)
# With forcing
f = Forms.AnalyticalFormField(
    1, x -> [ones(size(x, 1)), ones(size(x, 1))], Forms.get_geometry(B[1]), "f"
)
inputs = Assemblers.WeakFormInputs(B, B, (f,))
wf = with_rhs_weak_form(inputs, dΩ)
A, b = Assemblers.assemble(wf, bc)
for i in keys(bc)
    row = zeros(size(A, 2))
    row[i] = 1.0
    @test A[i, :] == row
    @test b[i] == bc[i]
end

# No forcing
inputs = Assemblers.WeakFormInputs(B, B)
wf = zero_rhs_weak_form(inputs, dΩ)
A, b = Assemblers.assemble(wf, bc)
for i in keys(bc)
    row = zeros(size(A, 2))
    row[i] = 1.0
    @test A[i, :] == row
    @test b[i] == bc[i]
end

# Matrix rhs
inputs = Assemblers.WeakFormInputs(B, B)
wf = zero_rhs_weak_form(inputs, dΩ)
full_A = copy(A)
A, b = Assemblers.assemble(wf, bc; rhs_type=Matrix{Float64})
non_bc = setdiff(1:size(full_A, 2), keys(bc))
@test A == full_A[non_bc, non_bc]
@test b == zeros(7, 7)

end
