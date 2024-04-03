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

function get_element_size(knot_vector::KnotVector, element_id::Int)
    return Mesh.get_element_size(knot_vector.patch_1d, element_id)
end



"""
    struct BSplineSpace

Structure containing information about a univariate B-Spline function space defined on `patch_1d::Mesh.Patch1D`,
with given `polynomial_degree` and `regularity` per breakpoint.

# Fields
- `knot_vector::KnotVector`: 1-dimensional knot vector.
- `extraction_op::ExtractionOperator`: Stores extraction coefficients and basis indices.
- `polynomials::ElementSpaces.Bernstein`: Refence Bernstein polynomials.
"""
struct BSplineSpace<:AbstractFiniteElementSpace{1}
    knot_vector::KnotVector
    extraction_op::ExtractionOperator
    polynomials::ElementSpaces.Bernstein
    
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

        new(knot_vector, extract_bspline_to_bernstein(knot_vector), ElementSpaces.Bernstein(polynomial_degree))
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
- `::ElementSpaces.Bernstein`: Bernstein polynomials.
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
    return size(bspline.knot_vector.patch_1d)*(bspline.knot_vector.polynomial_degree+1) + sum(bspline.knot_vector.multiplicity .- (bspline.knot_vector.polynomial_degree+1))
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
    el_size = get_element_size(bspline, element_id)
    for r = 0:nderivatives
        local_basis[:,:,r+1] .= @views local_basis[:,:,r+1] * extraction_coefficients
    end

    return local_basis, basis_indices
end

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