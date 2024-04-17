"""
    struct KnotVector

1-dimensional knot vector. 'patch_1d' determines the location of the knots and
'multiplicity' the number of times each of them is repeated.

# Fields
- `patch_1d::Mesh.Patch1D`: Location of each knot.
- `polynomial_degree::Int`: Polynomial degree.
- `multiplicity::Vector{Int}`: Repetition of each knot.
"""
struct KnotVector
    patch_1d::Mesh.Patch1D
    polynomial_degree::Int
    multiplicity::Vector{Int}

    function KnotVector(patch_1d::Mesh.Patch1D, polynomial_degree::Int, multiplicity::Vector{Int})
        if (size(patch_1d)+1) != length(multiplicity)
            msg1 = "Number of knots and multiplicity must be the same."
            msg2 = " Knots as length $(size(patch_1d)+1) and multiplicity has length $(length(multiplicity))."
            throw(ArgumentError(msg1*msg2))
        end

        if any(multiplicity .<= 0)
            error_index = findfirst(multiplicity .<= 0)
            msg1 = "multiplicity must be greater or equal to 1."
            msg2 = " In index $error_index, the multiplicity is $(multiplicity[error_index])."
            throw(ArgumentError(msg1*msg2))
        end

        new(patch_1d, polynomial_degree, multiplicity)
    end
end

"""
    create_knot_vector(patch_1d::Mesh.Patch1D, p::Int, regularity::Vector{Int})

Creates a uniform knot vector corresponding to B-splines basis functions of polynomial degree `p` 
and continuity `regularity[i]` on `patch_1d.breakpoint[i]`. 
See https://en.wikipedia.org/wiki/B-spline#Definition.

# Arguments
- `patch_1d::Mesh.Patch1D`: Location of each breakpoint.
- `p::Int`: degree of the polynomial (``p \\geq 0``).
- `breakpoint_condition::Vector{Int}`: Either the regularity or multiplicity of each breakpoint.
- `condition_type::String`: Determines whether `breakpoint_condition` determines the regularity or multiplicity.

# Returns 
- `::KnotVector`: Knot vector.
"""
function create_knot_vector(patch_1d::Mesh.Patch1D, p::Int, breakpoint_condition::Vector{Int}, condition_type::String)
    if condition_type == "regularity"
        multiplicity = ones(Int, size(patch_1d)+1)

        for i in eachindex(breakpoint_condition)
            multiplicity[i] = p - breakpoint_condition[i]
        end

        return KnotVector(patch_1d, p, multiplicity)
    elseif condition_type == "multiplicity"
        return KnotVector(patch_1d, p, vec)
    else 
        msg1 = "Allowed breakpoint conditions are `regularity` or `multiplicity`."
        msg2 = " The given condition is $condition_type."
        throw(ArgumentError(msg1*msg2))
    end
end

"""
    get_element_size(knot_vector::KnotVector, element_id::Int)

Returns the size of the element specified by `element_id`.

# Arguments
- `knot_vector::KnotVector`: The B-Spline function space.
- `element_id::Int`: The id of the element.
# Returns
- `::Float64`: The size of the element.
"""
function get_element_size(knot_vector::KnotVector, element_id::Int)
    return Mesh.get_element_size(knot_vector.patch_1d, element_id)
end

"""
    get_knot_length(knot_vector::KnotVector)

Determines the length of `knot_vector` by summing the multiplicites of each knot vector.

# Arguments
- `knot_vector::KnotVector`: Knot vector for length calculation.
# Returns
- `::Int`: Length of the knot vector.
"""
function get_knot_length(knot_vector::KnotVector)
    return sum(knot_vector.multiplicity)
end

"""
    get_breakpoint_index(knot_vector::KnotVector, full_index::Int)

Retrives the breakpoint index corresponding to `knot_vector` at `full_index`, i.e. the index 
of the vector where every `breakpoint[i]` appears `knot_vector.multiplicity[i]`-times.
# Arguments
- `knot_vector::KnotVector`: Knot vector for length calculation.
- `full_index::Int`: Index in the knot vector.
# Returns
- `index::Int`: Index of breakpoint corresponding to `full_index`.
"""
function get_breakpoint_index(knot_vector::KnotVector, full_index::Int)
    index = 1

    for i in 1:(size(knot_vector.patch_1d)+1)
        for _ in 1:knot_vector.multiplicity[i]
            if index == full_index
                return i
            end

            index += 1
        end
    end

    return error("Index out of bounds.")
end

function get_first_knot_index(knot_vector::KnotVector, breakpoint_index::Int)
    return cumsum(knot_vector.multiplicity)[breakpoint_index] - knot_vector.multiplicity[breakpoint_index] + 1
end

function get_last_knot_index(knot_vector::KnotVector, breakpoint_index::Int)
    return cumsum(knot_vector.multiplicity)[breakpoint_index]
end

"""
    get_knot_breakpoint(knot_vector::KnotVector, full_index::Int)

Retrives the breakpoint corresponding to `knot_vector` at `full_index`, i.e. the index 
of the vector where every `breakpoint[i]` appears `knot_vector.multiplicity[i]`-times.
# Arguments
- `knot_vector::KnotVector`: Knot vector for length calculation.
- `full_index::Int`: Index in the knot vector.
# Returns
- `::Float64`: Breakpoint corresponding to `full_index`.
"""
function get_knot_breakpoint(knot_vector::KnotVector, full_index::Int)
    index = get_breakpoint_index(knot_vector, full_index)
    return Mesh.get_breakpoints(knot_vector.patch_1d)[index]
end

"""
    get_knot_multiplicity(knot_vector::KnotVector, full_index::Int)

Retrives the multiplicity of the breakpoint corresponding to `knot_vector` at `full_index`, i.e. the index 
of the vector where every `breakpoint[i]` appears `knot_vector.multiplicity[i]`-times.
# Arguments
- `knot_vector::KnotVector`: Knot vector for length calculation.
- `full_index::Int`: Index in the knot vector.
# Returns
- `::Int`: Multiplicity of the breakpoint corresponding to `full_index`.
"""
function get_knot_multiplicity(knot_vector::KnotVector, full_index::Int)
    index = get_breakpoint_index(knot_vector, full_index)
    return knot_vector.multiplicity[index]
end

"""
    get_local_knot_vector(knot_vector::KnotVector, basis_id::Int)

Returns the local knot vector necessary to characterize the B-spline identified by `basis_id`.

# Arguments
- `knot_vector::KnotVector`: The knot vector of the full B-spline basis.
- `basis_id::Int`: The id of the B-spline.
# Returns
- `::KnotVector`: The knot vector of the B-spline identified by `basis_id`.
"""
function get_local_knot_vector(knot_vector::KnotVector, basis_id::Int)
    local_idx = get_breakpoint_index(knot_vector, basis_id):get_breakpoint_index(knot_vector, basis_id+knot_vector.polynomial_degree+1)

    local_patch = Mesh.Patch1D(Mesh.get_breakpoints(knot_vector.patch_1d)[local_idx])
    local_multiplicity = knot_vector.multiplicity[local_idx]

    return KnotVector(local_patch, knot_vector.polynomial_degree, local_multiplicity)
end

"""
    struct BSplineSpace

Structure containing information about a univariate B-Spline function space defined on `patch_1d::Mesh.Patch1D`,
with given `polynomial_degree` and `regularity` per breakpoint.

# Fields
- `knot_vector::KnotVector`: 1-dimensional knot vector.
- `extraction_op::ExtractionOperator`: Stores extraction coefficients and basis indices.
- `polynomials::Bernstein`: Refence Bernstein polynomials.
"""
struct BSplineSpace<:AbstractFiniteElementSpace{1}
    knot_vector::KnotVector
    extraction_op::ExtractionOperator
    polynomials::Bernstein
    
    function BSplineSpace(patch_1d::Mesh.Patch1D, polynomial_degree::Int, regularity::Vector{Int})
        # Check for errors in the construction 
        if polynomial_degree <0
            msg1 = "Polynomial degree must be greater or equal than 0."
            msg2 = " The degree is $polynomial_degree."
            throw(ArgumentError(msg1*msg2))
        end
        if (size(patch_1d)+1) != length(regularity)
            msg1 = "Number of regularity conditions should be equal to the number of breakpoints."
            msg2 = " You have $(size(patch_1d)+1) breakpoints and $(length(regularity)) regularity conditions."
            throw(ArgumentError(msg1*msg2))
        end
        for i in eachindex(regularity)
            if polynomial_degree <= regularity[i]
                msg1 = "Polynomial degree must be greater than regularity."
                msg2 = " The degree is $polynomial_degree and there is regularity $regularity[i] in index $i."
                throw(ArgumentError(msg1*msg2))
            end
        end
        
        knot_vector = create_knot_vector(patch_1d, polynomial_degree, regularity, "regularity")

        new(knot_vector, extract_bspline_to_bernstein(knot_vector), Bernstein(polynomial_degree))
    end
end

""" 
    get_extraction_coefficients_and_indices(bspline::BSplineSpace, element_id::Int)

Returns the supported bases functions of `bspline` on the element specified by `element_id` and their extraction coefficients.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `element_id::Int`: The id of the element.
# Returns
- `::@views (Matrix{Float64}, Vector{Int})`: The extraction coefficients and supported bases.
"""
function get_extraction(bspline::BSplineSpace, element_id::Int)
    return get_extraction(bspline.extraction_op, element_id)
end

"""
    get_polynomials(bspline::BSplineSpace)

Returns the reference Bernstein polynomials of `bspline`.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
# Returns
- `::Bernstein`: Bernstein polynomials.
"""
function get_local_basis(bspline::BSplineSpace, element_id::Int, xi::Vector{Float64}, nderivatives::Int)
    local_basis = bspline.polynomials(xi, nderivatives)
    el_size = get_element_size(bspline, element_id)
    for r = 0:nderivatives
        local_basis[:,:,r+1] .= @views local_basis[:,:,r+1] ./ el_size^r
    end
    
    return local_basis
end

"""
    get_local_knot_vector(bspline::BSplineSpace, basis_id::Int)

Returns the local knot vector necessary to characterize the B-spline identified by `basis_id`.

# Arguments
- `bspline::BSplineSpace`: The full B-spline basis.
- `basis_id::Int`: The id of the B-spline.
# Returns
- `::KnotVector`: The knot vector of the B-spline identified by `basis_id`.
"""
function get_local_knot_vector(bspline::BSplineSpace, basis_id::Int)
    return get_local_knot_vector(bspline.knot_vector, basis_id)
end

"""
    get_local_knot_vector(bspline::NTuple{n, BSplineSpace}, basis_id::NTuple{n, Int}) where {n}

Returns the local knot vector necessary to characterize the `n`-variate B-spline identified by `basis_id`.

# Arguments
- `bspline::NTuple{n, BSplineSpace}`: The full B-spline basis.
- `basis_id::NTuple{n, Int}`: The id of the B-spline.
# Returns
- `::NTuple{n, KnotVector}`: The knot vector of the B-spline identified by `basis_id`.
"""
function get_local_knot_vector(bspline::NTuple{n, BSplineSpace}, basis_id::NTuple{n, Int}) where {n}
    return ntuple(d -> get_local_knot_vector(bspline[d], basis_id[d]), n)
end

"""
    get_polynomial_degree(bspline::BSplineSpace)

Returns the degree of Bsplines

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
# Returns
- `polynomial_degree`: Polynomial degree.
"""
function get_polynomial_degree(bspline::BSplineSpace)
    return bspline.knot_vector.polynomial_degree
end

"""
    get_dim(bspline::BSplineSpace)

Returns the dimension of the univariate function space `bspline`.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.
# Returns
- `::Int`: The dimension of the B-Spline space.
"""
function get_dim(bspline::BSplineSpace)
    @assert get_dim(bspline.extraction_op) == size(bspline.knot_vector.patch_1d)*(bspline.knot_vector.polynomial_degree+1) + sum(bspline.knot_vector.multiplicity .- (bspline.knot_vector.polynomial_degree+1)) "B-spline dimension incorrect."
    return get_dim(bspline.extraction_op)
end

"""
    get_patch(bspline::BSplineSpace)

Returns the patch of the univariate function space `bspline`.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.
# Returns
- `::Mesh.Patch1D`: The patch of the B-Spline space.
"""
function get_patch(bspline::BSplineSpace)
    return bspline.knot_vector.patch_1d
end

"""
    get_patch(bspline::BSplineSpace)

Returns the multiplicity of the knot vector associated with the univariate function space `bspline`.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.
# Returns
- `::Vector{Int}`: The multiplicity of the knot vector associated with the B-Spline space.
"""
function get_multiplicity(bspline::BSplineSpace)
    return bspline.knot_vector.multiplicity
end

"""
    get_num_elements(bspline::BSplineSpace)

Returns the number of elements in the underlying partition.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.
# Returns
- `::Int`: The number of elements.
"""
function get_num_elements(bspline::BSplineSpace)
    return size(bspline.knot_vector.patch_1d)
end

"""
    get_element_size(bspline::BSplineSpace, element_id::Int)

Returns the size of the element specified by `element_id`.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.
- `element_id::Int`: The id of the element.
# Returns
- `::Float64`: The size of the element.
"""
function get_element_size(bspline::BSplineSpace, element_id::Int)
    return get_element_size(bspline.knot_vector, element_id)
end

"""
    evaluate(bspline::BSplineSpace, element_id::Int, xi::Vector{Float64}, nderivatives::Int)

Evaluates the non-zero `bspline` basis functions on the element specified by `element_id` and points `xi` and all derivatives up to nderivatives.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `element_id::Int`: The id of the element.
- `xi::Vector{Float64}`: The points where the global basis is evaluated.
- `nderivatives::Int`: The order upto which derivatives need to be computed.
# Returns
- `::Array{Float64}`: Global basis functions, size = n_eval_points x degree+1 x nderivatives+1
"""
function evaluate(bspline::BSplineSpace, element_id::Int, xi::Vector{Float64}, nderivatives::Int)
    extraction_coefficients, basis_indices = get_extraction(bspline, element_id)
    local_basis = get_local_basis(bspline, element_id, xi, nderivatives)
    for r = 0:nderivatives
        local_basis[:,:,r+1] .= @views local_basis[:,:,r+1] * extraction_coefficients
    end

    return local_basis, basis_indices
end

"""
    evaluate(bspline::BSplineSpace, element_id::Int, xi::Float64, nderivatives::Int)

Evaluates the non-zero `bspline` basis functions on the element specified by `element_id` on point `xi` and all derivatives up to nderivatives.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `element_id::Int`: The id of the element.
- `xi::Float64`: The point where the global basis is evaluated.
- `nderivatives::Int`: The order upto which derivatives need to be computed.
# Returns
- `::Array{Float64}`: Global basis functions, size = 1 x degree+1 x nderivatives+1
"""
function evaluate(bspline::BSplineSpace, element_id::Int, xi::Float64, nderivatives::Int)
    return evaluate(bspline, element_id, [xi], nderivatives)
end

"""
evaluate_all_at_point(bspline::BSplineSpace, element_id::Int, xi::Float64, nderivatives::Int)

Evaluates all derivatives upto order `nderivatives` for all `bspline` basis functions at a given point `xi` in the element `element_id`.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `element_id::Int`: The id of the element.
- `xi::Float64`: The point where all global basis functiuons are evaluated.
- `nderivatives::Int`: The order upto which derivatives need to be computed.
# Returns
- `::SparseMatrixCSC{Float64}`: Global basis functions, size = n_dofs x nderivatives+1
"""
function evaluate_all_at_point(bspline::BSplineSpace, element_id::Int, xi::Float64, nderivatives::Int)
    local_basis, basis_indices = evaluate(bspline, element_id, xi, nderivatives)
    nloc = length(basis_indices)
    ndofs = get_dim(bspline)
    I = zeros(Int, nloc * (nderivatives + 1))
    J = zeros(Int, nloc * (nderivatives + 1))
    V = zeros(Float64, nloc * (nderivatives + 1))
    count = 0
    for r = 0:nderivatives
        for i = 1:nloc
            I[count+1] = basis_indices[i]
            J[count+1] = r+1
            V[count+1] = local_basis[1, i, r+1]
            count += 1
        end
    end

    return SparseArrays.sparse(I,J,V,ndofs,nderivatives+1)
end

"""
    evaluate(bspline::BSplineSpace, element_id::Int, xi::Vector{Float64}, nderivatives::Int, coefficients::Vector{Float64})

Evaluates a spline on the element specified by `element_id` and points `xi` and all derivatives up to nderivatives, form a 
`bspline` basis with given `coefficients`.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `element_id::Int`: The id of the element.
- `xi::Vector{Float64}`: The points where the global basis is evaluated.
- `nderivatives::Int`: The order upto which derivatives need to be computed.
- `coefficients::Vector{Float64}`: Coefficients of the spline with basis `bspline`.
# Returns
- `::Array{Float64}`: Spline evaluation (size = n_eval_points x nderivatives+1).
"""
function evaluate(bspline::BSplineSpace, element_id::Int, xi::Vector{Float64}, nderivatives::Int, coefficients::Vector{Float64})
    local_basis, basis_indices = evaluate(bspline, element_id, xi, nderivatives)
    evaluation = zeros(Float64, (size(local_basis)[1],nderivatives+1) )
    
    for r = 0:nderivatives
        evaluation[:,r+1] .= @views sum(local_basis[:,:,r+1] .* coefficients[basis_indices]', dims=2)
    end

    return evaluation
end

"""
    GTBSplineSpace constructor
"""
function GTBSplineSpace(bsplines::NTuple{m,BSplineSpace}, regularity::Vector{Int}) where {m}
    if length(regularity) != m
        msg1 = "Number of regularity conditions should be equal to the number of bspline interfaces."
        msg2 = " You have $(m) interfaces and $(length(regularity)) regularity conditions."
        throw(ArgumentError(msg1*msg2))
    end
    for i in 1:m
        j = i
        k = i+1
        if i == m
            k = 1
        end
        polynomial_degree = min(get_polynomial_degree(bsplines[j]), get_polynomial_degree(bsplines[k]))
        if polynomial_degree < regularity[i]
            msg1 = "Minimal polynomial degrees must be greater than or equal to the regularity."
            msg2 = " The minimal degree is $polynomial_degree and there is regularity $regularity[i] in index $i."
            throw(ArgumentError(msg1*msg2))
        end
    end
    return UnstructuredSpace(bsplines, extract_gtbspline_to_bspline(bsplines, regularity), Dict("regularity" => regularity))
end