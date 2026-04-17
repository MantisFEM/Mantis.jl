
include("BernsteinPolynomials.jl")
include("LagrangePolynomials.jl")
include("ECTSpaces/ECTSpaces.jl")

"""
    get_polynomial_degree(elem_loc_basis::AbstractCanonicalSpace)

Returns the polynomial degree of the element-local basis.

# Arguments
- `elem_loc_basis::AbstractCanonicalSpace`: An element-local basis.

# Returns
- `::Int`: The polynomial degree of the element-local basis.
"""
function get_polynomial_degree(elem_loc_basis::AbstractCanonicalSpace)
    return elem_loc_basis.p
end

"""
    get_derivative_space(elem_loc_basis::AbstractCanonicalSpace)

This default method returns the element-local basis of one degree lower than the given
element-local basis. This method should be overloaded for element-local bases that do not
satisfy this property or those that need additional parameters; e.g., ECT spaces.

# Arguments
- `elem_loc_basis::AbstractCanonicalSpace`: An element-local basis.

# Returns
- `::AbstractCanonicalSpace`: The element-local basis of one degree lower than the given
    element-local basis.
"""
function get_derivative_space(elem_loc_basis::AbstractCanonicalSpace)
    return typeof(elem_loc_basis)(max(elem_loc_basis.p - 1, 0))
end

"""
    get_bisected_canonical_space(elem_loc_basis::AbstractCanonicalSpace)

This default method returns the given element-local basis. This method should be overloaded
for element-local bases that do not satisfy this property or those that need additional
parameters; e.g., ECT spaces.

# Arguments
- `elem_loc_basis::AbstractCanonicalSpace`: An element-local basis.

# Returns
- `::AbstractCanonicalSpace`: The input element-local basis.
"""
function get_bisected_canonical_space(elem_loc_basis::AbstractCanonicalSpace)
    return elem_loc_basis
end

"""
    get_child_canonical_space(elem_loc_basis::AbstractCanonicalSpace, num_sub_elements::Int)

This default method returns the given element-local basis. This method should be overloaded
for element-local bases that do not satisfy this property or those that need additional
parameters; e.g., ECT spaces.

# Arguments
- `elem_loc_basis::AbstractCanonicalSpace`: An element-local basis.
- `num_sub_elements::Int`: The number of sub-elements to divide the canonical space into.

# Returns
- `::AbstractCanonicalSpace`: The input element-local basis.
"""
function get_child_canonical_space(
    elem_loc_basis::AbstractCanonicalSpace, num_sub_elements::Int
)
    return elem_loc_basis
end

"""
    _evaluate_all_at_point(
        canonical_space::AbstractCanonicalSpace, xi::Float64, nderivatives::Int
    )

Evaluates all derivatives upto order `nderivatives` for all basis functions of
`canonical_space` at a given point `xi`.

# Arguments
- `canonical_space::AbstractCanonicalSpace`: A canonical space.
- `xi::Float64`: The point where all global basis functiuons are evaluated.
- `nderivatives::Int`: The order upto which derivatives need to be computed.

# Returns
- `::SparseMatrixCSC{Float64}`: Global basis functions, size = n_dofs x nderivatives+1
"""
function _evaluate_all_at_point(
    canonical_space::AbstractCanonicalSpace, xi::Float64, nderivatives::Int
)
    local_basis = evaluate(canonical_space, Points.CartesianPoints(([xi],)), nderivatives)
    ndofs = get_polynomial_degree(canonical_space) + 1
    basis_indices = collect(1:ndofs)
    I = zeros(Int, ndofs * (nderivatives + 1))
    J = zeros(Int, ndofs * (nderivatives + 1))
    V = zeros(Float64, ndofs * (nderivatives + 1))
    count = 0
    for r in 0:nderivatives
        for i in 1:ndofs
            I[count + 1] = basis_indices[i]
            J[count + 1] = r + 1
            V[count + 1] = local_basis[r + 1][1][1, i]
            count += 1
        end
    end

    return SparseArrays.sparse(I, J, V, ndofs, nderivatives + 1)
end

"""
    build_two_scale_matrix(canonical_space::AbstractCanonicalSpace, num_sub_elements::Int)

Uniformly subdivides the canonical space into `num_sub_elements` sub-elements. It is
assumed that `num_sub_elements` is a power of 2, else the method throws an argument error.
It returns a global subdivision matrix that maps the global basis functions of the
canonical space to the global basis functions of the subspaces.

# Arguments
- `canonical_space::AbstractCanonicalSpace`: A canonical space.
- `num_sub_elements::Int`: The number of subspaces to divide the canonical space into.

# Returns
- `::SparseMatrixCSC{Float64}`: A global subdivision matrix that maps the global basis
    functions of the canonical space to the global basis functions of the subspaces.
"""
function build_two_scale_matrix(
    canonical_space::AbstractCanonicalSpace, num_sub_elements::Int
)
    num_ref = log2(num_sub_elements)
    if num_sub_elements < 2 || !isapprox(num_ref - round(num_ref), 0.0; atol=1e-12)
        throw(
            ArgumentError(
                "Number of subdivisions should be a power of 2 and greater than 1"
            ),
        )
    end
    p = get_polynomial_degree(canonical_space)
    num_ref = Int(num_ref)

    # get bisected canonical space
    bisected_canonical_space = get_bisected_canonical_space(canonical_space)
    # evaluate points on the finer elements
    ξ = Points.PointSet((LinRange(0.0, 1.0, p + 1),))
    ξ_L = Points.PointSet((LinRange(0.0, 0.5, p + 1),))
    ξ_R = Points.PointSet((LinRange(0.5, 1.0, p + 1),))
    # evaluate all fine basis functions at the Greville points
    fine_eval = evaluate(bisected_canonical_space, ξ)[1][1]
    # evaluate all coarse basis functions on the left and right elements
    coarse_eval_L = evaluate(canonical_space, ξ_L)[1][1]
    coarse_eval_R = evaluate(canonical_space, ξ_R)[1][1]
    # bisection matrix
    bisection_matrix = SparseArrays.sparse(
        [
            fine_eval \ coarse_eval_L
            fine_eval \ coarse_eval_R
        ]
    )

    # now, build the subdivision matrix for num_sub_elements > 2
    subdivision_matrix = SparseArrays.sparse(Matrix(LinearAlgebra.I, p + 1, p + 1))
    for i in 1:num_ref
        subdivision_matrix =
            SparseArrays.blockdiag([bisection_matrix for i in 1:(2^(i - 1))]...) * subdivision_matrix
    end

    return subdivision_matrix
end
