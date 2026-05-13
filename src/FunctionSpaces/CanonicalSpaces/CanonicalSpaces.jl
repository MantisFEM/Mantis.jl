
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
    get_canonical_space_on_subelements(space::AbstractCanonicalSpace; num_subdivisions::Int = 2, degree_delta::Int = 0)

When a canonical element is uniformly subdivided into `num_subdivisions` sub-elements,
this function returns the canonical space which can be used on all sub-elements. The method
also increases the polynomial degree of the fine spaces by `degree_delta`.

# Arguments
- `space::AbstractCanonicalSpace`: A canonical space.
- `num_subdivisions::Int`: The number of sub-elements to divide the canonical element into.
- `degree_delta::Int`: The increase in polynomial degree for the fine spaces.

# Returns
- `::AbstractCanonicalSpace`: The canonical space on the sub-elements.
"""
function get_canonical_space_on_subelements(
    space::AbstractCanonicalSpace; num_subdivisions::Int = 2, degree_delta::Int = 0
)   
    return typeof(space)(space.p + degree_delta)
end

"""
    build_two_scale_matrix(space::AbstractCanonicalSpace; 
        num_subdivisions::Int = 2, degree_delta::Int = 0)

Returns a global two-scale matrix for a given canonical space. This matrix maps the global 
coefficients of the parent basis to the global coefficients of the children basis.

The finer spaces are constructed by uniformly subdividing the parametric domain of the 
parent canonical space into `num_subdivisions` subspaces of equal size and increasing the 
polynomial degree of the fine spaces by `degree_delta`. 

# Arguments
- `space::AbstractCanonicalSpace`: A canonical space.
- `num_subdivisions::Int`: The number of elements to divide the canonical element into.
- `degree_delta::Int`: The increase in polynomial degree for the fine spaces.

# Returns
- `::SparseMatrixCSC{Float64}`: A global subdivision matrix that maps the global coefficients
    of the parent basis to the global coefficients of the children basis.
"""
function build_two_scale_matrix(
    space::AbstractCanonicalSpace; num_subdivisions::Int = 2, degree_delta::Int = 0
)
    p = get_polynomial_degree(space)
    if num_subdivisions == 1 && degree_delta == 0
        return SparseArrays.sparse(Matrix(LinearAlgebra.I, p + 1, p + 1))
    end

    # uniform subdivision breakpoints
    breakpoints = LinRange(0.0, 1.0, num_subdivisions + 1)

    # build all fine spaces
    space_fine = get_canonical_space_on_subelement(
        space; num_subdivisions = num_subdivisions, degree_delta = degree_delta
    )

    # evaluation points for the finer spaces w.r.t. each sub-element
    ξ_fine = Points.PointSet((LinRange(0.0, 1.0, p + degree_delta + 1),))

    # evaluation points for the coarse space w.r.t. entire element
    ξ_coarse = [
        Points.PointSet((LinRange(breakpoints[i], breakpoints[i + 1], p + degree_delta + 1),))
        for i in 1:num_subdivisions
    ]
    
    # evaluate all fine basis functions
    fine_eval = evaluate(space_fine, ξ_fine)[1][1]
    # evaluate all coarse basis functions w.r.t. entire element
    coarse_evals = [evaluate(space, ξ_coarse[i])[1][1] for i in 1:num_subdivisions]
    # subdivision matrix obtained by solving `num_subdivisions` linear systems
    subdivision_matrix = SparseArrays.sparse(
        vcat([fine_eval \ coarse_evals[i] for i in 1:num_subdivisions]...)
    )
    return subdivision_matrix
end