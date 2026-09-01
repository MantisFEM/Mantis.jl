
function Hierarchical.build_scaling_matrix(
    parent::AbstractFESpace, child::AbstractFESpace, method::Function
)
    return method(parent, child)
end

############################################################################################
#                                         Uniform                                          #
############################################################################################

function _check_degrees(parent, child)
    parent_p = get_polynomial_degree(parent)
    child_p = get_polynomial_degree(child)
    if parent_p != child_p
        throw(
            ArgumentError(
                LazyString(
                    "Parent and child polynomial degrees must match. ",
                    "Got ",
                    parent_p,
                    " and ",
                    child_p,
                    ".",
                ),
            ),
        )
    end

    return nothing
end

function scaling_matrix_uniform(
    parent::BSplineSpace{Bernstein}, child::BSplineSpace{Bernstein}
)
    _check_degrees(parent, child)

    return scaling_matrix_uniform(get_knot_vector(parent), get_knot_vector(child))
end

function scaling_matrix_uniform(parent::KnotVector, child::KnotVector)
    _check_degrees(parent, child)

    m = get_knot_vector_length(child)
    p = get_polynomial_degree(parent)
    nchild = m - p - 1

    gm_values = zeros(nchild * (p + 1))
    gm_rows = zeros(Int, nchild * (p + 1))
    gm_columns = similar(gm_rows)

    cf = p + 1
    rf = 1
    e = 1

    # Two-copies of pre-allocated arrays used in the algorithm
    tmp_1 = Matrix{Float64}(LinearAlgebra.I, p + 1, p + 1)
    tmp_2 = deepcopy(tmp_1)
    while rf <= nchild
        mult = get_knot_multiplicity(child, rf)

        lastcf = cf
        while get_knot_value(parent, cf + 1) <= get_knot_value(child, rf)
            cf += 1
        end

        if e > 1
            offs = cf - lastcf
            tmp_2[1:(p + 1 - offs), 1:(p + 1 - mult)] .= view(
                tmp_1, (1 + offs):(p + 1), (1 + mult):(p + 1)
            )
        end

        for t in (p + 2 - mult):(p + 1)
            rng = ((rf - 1) * (p + 1) + 1):(rf * (p + 1))
            gm_columns[rng] .= (cf - p):cf
            gm_rows[rng] .= rf

            tmp_2[:, t] = single_knot_oslo(parent, child, cf, rf)
            gm_values[rng] .= view(tmp_2, :, t)

            rf += 1
        end

        e += 1
        tmp_1, tmp_2 = tmp_2, tmp_1
    end

    scaling_matrix = SparseArrays.dropzeros(
        SparseArrays.sparse(gm_rows, gm_columns, gm_values, rf - 1, cf)
    )

    return scaling_matrix
end

function scaling_matrix_uniform(space::Bernstein, num_subdivisions::Int)
    p = get_polynomial_degree(space)
    parent_knot_vector = KnotVector(
        Geometry.CartesianGeometry(LinRange(0, 1, 2)), p, [p+1, p+1]
    )
    child_knot_vector = refinement_uniform(parent_knot_vector, num_subdivisions, p + 1)

    return scaling_matrix_uniform(parent_knot_vector, child_knot_vector)
end

"""
    single_knot_oslo(
        parent_knot_vector::KnotVector, child_knot_vector::KnotVector, cf::Int, rf::Int
    )

Algorithm for the coefficients of a change of B-spline representation for a single knot
insertion. The parent knot vector is `parent_knot_vector` and the inserted knot is given by
`child_knot_vector`.

For more information, see
[A note on the Oslo Algorithm](https://collections.lib.utah.edu/dl_files/66/d4/66d493df0f5c97cce67e0bc1294363d64dde7f06.pdf).

# Arguments
- `parent_knot_vector::KnotVector`: parent knot vector.
- `child_knot_vector::KnotVector`: child knot vector, with the extra knot.
- `cf::Int`: Index of the parent knot vector.
- `rf::Int`: Index of the child knot vector such that
    `get_knot_value(parent_knot_vector,cf) <= get_knot_value(child_knot_vector,rf) <
    get_knot_value(parent_knot_vector,cf+1)`.

# Returns
- `b::Vector{Float64}`: Coefficients for the change of basis.
"""
function single_knot_oslo(
    parent_knot_vector::KnotVector, child_knot_vector::KnotVector, cf::Int, rf::Int
)
    p = get_polynomial_degree(parent_knot_vector)
    b = zeros(p + 1)
    b[1] = 1
    tmp = similar(b)
    for k in 1:p
        fill!(tmp, 0.0)
        x = get_knot_value(child_knot_vector, rf + k)
        for j in 1:k
            t1 = get_knot_value(parent_knot_vector, cf + j - k)
            t2 = get_knot_value(parent_knot_vector, cf + j)
            w = (x - t1) / (t2 - t1)
            tmp[j] += (1 - w) * b[j]
            tmp[j + 1] += w * b[j]
        end

        b, tmp = tmp, b
    end

    return b
end

function scaling_matrix_uniform(parent::T, child::T) where {T <: TensorProductSpace}
    parent_factors = get_factor_spaces(parent)
    child_factors = get_factor_spaces(child)
    matrix_factors = map(scaling_matrix_uniform, parent_factors, child_factors)

    return LinearAlgebra.kron(Iterators.reverse(matrix_factors)...)
end

function scaling_uniform(
    parent::AbstractFESpace{manifold_dim}, num_subdivisions::NTuple{manifold_dim, Int}
) where {manifold_dim}
    child = refinement_uniform(parent, num_subdivisions)

    return MatrixScaling(parent, child, scaling_matrix_uniform)
end

############################################################################################
#                                     Degree-Elevation                                     #
############################################################################################

function scaling_matrix_degree(parent::BSplineSpace, child::BSplineSpace, degree_delta::Int)
    num_elements = get_num_elements(parent)
    if num_elements != get_num_elements(child)
        throw(
            ArgumentError(
                LazyString(
                    "Number of elements must match between the parent and child spaces. ",
                    "Got ",
                    num_elements,
                    " and ",
                    get_num_elements(child),
                    ".",
                ),
            ),
        )
    end
    #=
    This problem can be solved by locally solving a least-squares problem. So, we loop over
    each element and solve:
        E_c^e S^e = R E_p^e,
    where ^e is the current element, E_c and E_p are the child and parent local Bezier
    extractions, R is the degree elevation matrix (same for every element), and S^e is the
    scaling matrix. Finally, we assemble the full matrix with element-local matrices.
    =#
    R = scaling_matrix_degree(get_polynomials(parent), degree_delta)
    # pre-allocated arrays for element-local computations
    rhs = similar(R)
    pp = get_polynomial_degree(parent)
    pc = get_polynomial_degree(child)
    Ec_qr = Matrix{Float64}(undef, pc + 1, pc + 1)
    S = similar(R)
    # Matrix size (m, n)
    m = get_num_basis(child)
    n = get_num_basis(parent)
    # Pre-allocate sparse array storage
    nnz = num_elements * (pc + 1) * (pp + 1)
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{Float64}(undef, nnz)
    k = 0
    @inbounds for element in 1:num_elements
        Ec = get_extraction_coefficients(child, element)
        Ep = get_extraction_coefficients(parent, element)
        # rhs = R E_p^e
        LinearAlgebra.mul!(rhs, R, Ep)
        # Solve E_c^e * S^e = rhs
        copyto!(Ec_qr, Ec)
        lhs = LinearAlgebra.qr!(Ec_qr)
        LinearAlgebra.ldiv!(S, lhs, rhs)

        Ip = get_basis_indices(parent, element)
        Ic = get_basis_indices(child, element)
        for j in eachindex(Ip), i in eachindex(Ic)
            val = S[i, j]
            if !iszero(val)
                k += 1
                I[k] = Ic[i]
                J[k] = Ip[j]
                V[k] = val
            end
        end
    end

    # Remove unused nnz
    I = I[1:k]
    J = J[1:k]
    V = V[1:k]

    scaling_matrix = SparseArrays.sparse(I, J, V, m, n, (x, _) -> x) # To avoid duplication.
    SparseArrays.fkeep!((_, _, v) -> abs(v) > 1e-15, scaling_matrix)

    return scaling_matrix
end

function scaling_matrix_degree(parent::Bernstein, degree_delta::Int)
    degree_delta >= 0 || throw(
        ArgumentError(
            LazyString("`degree_delta` must be non-negative. Got ", degree_delta, ".")
        ),
    )
    p = get_polynomial_degree(parent)
    iszero(degree_delta) && return Matrix{Float64}(LinearAlgebra.I, p + 1, p + 1)

    scaling_matrix = zeros(p + degree_delta + 1, p + 1)
    S = _scaling_single_degree(p)
    isone(degree_delta) && (return S)

    curr = similar(scaling_matrix)
    copyto!(view(curr, 1:(p + 2), 1:(p + 1)), S)
    for d in 1:(degree_delta - 1)
        S = _scaling_single_degree(p + d)
        LinearAlgebra.mul!(
            view(scaling_matrix, 1:(p + d + 2), :), S, view(curr, 1:(p + d + 1), 1:(p + 1))
        )
        copyto!(
            view(curr, 1:(p + d + 2), 1:(p + 1)),
            view(scaling_matrix, 1:(p + d + 2), 1:(p + 1)),
        )
    end

    return scaling_matrix
end

function _scaling_single_degree(p::Int)
    p1 = p + 1
    scaling_matrix = Matrix{Float64}(LinearAlgebra.I, p1 + 1, p1)
    for j in 0:p
        scaling_matrix[j + 1, j + 1] = (p1 - j) / p1
        scaling_matrix[j + 2, j + 1] = (j + 1) / p1
    end

    return scaling_matrix
end

############################################################################################
#                                       Approximate                                        #
############################################################################################

function scaling_matrix_approximate(
    parent_space::AbstractFESpace{manifold_dim},
    child_space::AbstractFESpace{manifold_dim},
    canonical_scaling_matrix::AbstractMatrix{T};
    tol=1e-14,
) where {manifold_dim, T <: Real}
    if get_num_basis(child_space) < get_num_basis(parent_space)
        return throw(
            ArgumentError(
                "The child space must have more degrees of freedom than the parent space."
            ),
        )
    end

    parent_num_elements = get_num_elements(parent_space)
    canonical_scaling_matrices = fill(
        SparseArrays.sparse(canonical_scaling_matrix), parent_num_elements
    )

    return scaling_matrix_approximate(
        parent_space, child_space, canonical_scaling_matrices; tol=tol
    )
end

function scaling_matrix_approximate(
    parent_space::AbstractFESpace{manifold_dim},
    child_space::AbstractFESpace{manifold_dim},
    canonical_scaling_matrices::AbstractVector{<:AbstractMatrix{T}};
    tol=1e-14,
) where {manifold_dim, T <: Real}
    if get_num_basis(child_space) < get_num_basis(parent_space)
        return throw(
            ArgumentError(
                "The child space must have more degrees of freedom than the parent space."
            ),
        )
    end

    parent_num_elements = get_num_elements(parent_space)
    if parent_num_elements != length(canonical_scaling_matrices)
        return throw(
            ArgumentError(
                LazyString(
                    "Incorrect number of canonical scaling matrices: expected ",
                    parent_num_elements,
                    " (the number of parent elements), got ",
                    length(canonical_scaling_matrices),
                ),
            ),
        )
    end

    parent_global_extraction = assemble_global_extraction_matrix(parent_space)
    child_global_extraction = assemble_global_extraction_matrix(child_space)
    R = SparseArrays.blockdiag(canonical_scaling_matrices...)

    scaling_matrix = child_global_extraction \ Matrix(R * parent_global_extraction)
    z = zero(eltype(scaling_matrix))
    map!(v -> abs(v) < tol ? z : v, scaling_matrix)

    return scaling_matrix
end
