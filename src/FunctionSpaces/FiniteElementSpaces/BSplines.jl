@doc raw"""
    BSplineSpace{F} <: AbstractFiniteElementSpace{1}

Structure containing information about a univariate B-Spline function space defined on `patch_1d::Mesh.Patch1D`,
with given `polynomial_degree` and `regularity` per breakpoint. Note that while the section spaces on each element are the same, they don't necessarily have to be polynomials; they are just named `polynomials` for convention.

# Fields
- `knot_vector::KnotVector`: 1-dimensional knot vector.
- `extraction_op::ExtractionOperator`: Stores extraction coefficients and basis indices.
- `polynomials::F`: local section space F, named `polynomials` just for convention.
- `dof_partition::Vector{Vector{Vector{Int}}}`: Indices of boundary degrees of freedom.
"""
struct BSplineSpace{F} <: AbstractFiniteElementSpace{1}
    knot_vector::KnotVector
    extraction_op::ExtractionOperator
    polynomials::F
    dof_partition::Vector{Vector{Vector{Int}}}

    function BSplineSpace(patch_1d::Mesh.Patch1D, polynomials::F, regularity::Vector{Int}, n_dofs_left::Int, n_dofs_right::Int) where {F <: AbstractCanonicalSpace}
        # polynomial degree
        polynomial_degree = get_polynomial_degree(polynomials)

        # Check for errors in the construction 
        if polynomial_degree <0
            msg1 = "Polynomial degree must be greater or equal than 0."
            msg2 = " The degree is $(polynomial_degree)."
            throw(ArgumentError(msg1*msg2))
        end

        if (size(patch_1d) + 1) != length(regularity)
            msg1 = "Number of regularity conditions should be equal to the number of breakpoints."
            msg2 = " You have $(size(patch_1d) + 1) breakpoints and $(length(regularity)) regularity conditions."
            throw(ArgumentError(msg1 * msg2))
        end

        for i in eachindex(regularity)
            if polynomial_degree <= regularity[i]
                msg1 = "Polynomial degree must be greater than regularity."
                msg2 = " The degree is $polynomial_degree and there is regularity $regularity[i] in index $i."
                throw(ArgumentError(msg1 * msg2))
            end
        end

        if F <: AbstractLagrangePolynomials
            if maximum(regularity) > 0
                throw(ArgumentError("Smoothness for a piecewise-Lagrange space must be at most C^0."))
            end
        end

        # Create the knot vector
        knot_vector = create_knot_vector(patch_1d, polynomial_degree, regularity, "regularity")

        # Extract B-spline to local section space
        extraction_op = extract_bspline_to_section_space(knot_vector, polynomials)

        # Get the dimension of the B-spline space
        bspline_dim = get_num_basis(extraction_op)

        # Allocate memory for degree of freedom partitioning for this single patch
        dof_partition = Vector{Vector{Vector{Int}}}(undef,1)
        dof_partition[1] = Vector{Vector{Int}}(undef,3)
        # First, store the left dofs ...
        dof_partition[1][1] = collect(1:n_dofs_left)
        # ... then the interior dofs ...
        dof_partition[1][2] = collect(n_dofs_left+1:bspline_dim-n_dofs_right)
        # ... and then finally the right dofs.
        dof_partition[1][3] = collect(bspline_dim-n_dofs_right+1:bspline_dim)
        
        # Initialize the BSplineSpace struct
        new{F}(knot_vector, extraction_op, polynomials, dof_partition)
    end

    # constructor with default dof_partitioning
    function BSplineSpace(patch_1d::Mesh.Patch1D, polynomials::F, regularity::Vector{Int}) where {F <: AbstractCanonicalSpace}
        BSplineSpace(patch_1d, polynomials, regularity, 1, 1)
    end

    # constructor with default polynomials
    function BSplineSpace(patch_1d::Mesh.Patch1D, polynomial_degree::Int, regularity::Vector{Int})
        BSplineSpace(patch_1d, Bernstein(polynomial_degree), regularity, 1, 1)
    end

    # constructor with default internal regularity and dof_partitioning
    function BSplineSpace(patch_1d::Mesh.Patch1D, polynomials::F, regularity::Int) where {F <: AbstractCanonicalSpace}
        BSplineSpace(patch_1d, polynomials, [-1; repeat([regularity],size(patch_1d)-1); -1], 1, 1)
    end

    # constructor with default internal regularity, polynomials, and dof_partitioning
    function BSplineSpace(patch_1d::Mesh.Patch1D, polynomial_degree::Int, regularity::Int)
        BSplineSpace(patch_1d, Bernstein(polynomial_degree), [-1; repeat([regularity],size(patch_1d)-1); -1], 1, 1)
    end

end

"""
    get_extraction(bspline::BSplineSpace, element_id::Int)

Returns the supported basis functions of `bspline` on the element specified by `element_id` and their extraction coefficients.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `element_id::Int`: The id of the element.

# Returns
- `::Tuple{Matrix{Float64}, Vector{Int}}`: The extraction coefficients and supported bases.
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
function get_polynomials(bspline::BSplineSpace)
    return bspline.polynomials
end

"""
    get_local_basis(bspline::BSplineSpace, element_id::Int, xi::Vector{Float64}, nderivatives::Int)

Compute the local basis functions and their derivatives for a B-spline element.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `element_id::Int`: The id of the element.
- `xi::Vector{Float64}`: Evaluation points.
- `nderivatives::Int`: Number of derivatives to compute.

# Returns
- `::Matrix{Float64}`: Local basis functions and their derivatives.
"""
function get_local_basis(bspline::BSplineSpace, ::Int, xi::Vector{Float64}, nderivatives::Int)
    local_basis = evaluate(bspline.polynomials, xi, nderivatives)

    return local_basis
end

"""
    get_local_basis(bspline::BSplineSpace, element_id::Int, xi::NTuple{1, Vector{Float64}}, nderivatives::Int)

Compute the local basis functions and their derivatives for a B-spline element.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `element_id::Int`: The id of the element.
- `xi::NTuple{1, Vector{Float64}}`: Evaluation points.
- `nderivatives::Int`: Number of derivatives to compute.

# Returns
- `::Matrix{Float64}`: Local basis functions and their derivatives.
"""
function get_local_basis(bspline::BSplineSpace, element_id::Int, xi::NTuple{1, Vector{Float64}}, nderivatives::Int)
    return get_local_basis(bspline, element_id, xi[1], nderivatives)
end



"""
    get_polynomial_degree(bspline::BSplineSpace)

Returns the degree of B-splines.

# Arguments
- `bspline::BSplineSpace`: A univariate B-Spline function space.
- `elem_id::Int`: An optional dummy argument for uniformity with other spaces (dummy because the degree is the same for all elements for B-splines).

# Returns
- `polynomial_degree`: Polynomial degree.
"""
function get_polynomial_degree(bspline::BSplineSpace, elem_id::Int = 0)
    return get_polynomial_degree(bspline.polynomials)
end

"""
    get_num_basis(bspline::BSplineSpace)

Returns the dimension of the univariate function space `bspline`.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.

# Returns
- `::Int`: The dimension of the B-Spline space.
"""
function get_num_basis(bspline::BSplineSpace)
    @assert get_num_basis(bspline.extraction_op) == size(bspline.knot_vector.patch_1d) * (bspline.knot_vector.polynomial_degree + 1) + sum(bspline.knot_vector.multiplicity .- (bspline.knot_vector.polynomial_degree + 1)) "B-spline dimension incorrect."
    return get_num_basis(bspline.extraction_op)
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
    get_multiplicity_vector(bspline::BSplineSpace)

Returns the multiplicities of the knot vector associated with the univariate function space `bspline`.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.

# Returns
- `::Vector{Int}`: The multiplicity of the knot vector associated with the B-Spline space.
"""
function get_multiplicity_vector(bspline::BSplineSpace)
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
    get_support(bspline::BSplineSpace, basis_id::Int)

Returns the elements where the B-spline given by `basis_id` is supported.

# Arguments
- `bspline::BSplineSpace`: The B-Spline function space.
- `basis_id::Int`: The id of the basis function.

# Returns
- `::UnitRange{Int}`: The support of the basis function.
"""
function get_support(bspline::BSplineSpace, basis_id::Int)
    first_element = convert_knot_to_breakpoint_idx(bspline.knot_vector, basis_id)
    last_element = convert_knot_to_breakpoint_idx(bspline.knot_vector, basis_id + bspline.knot_vector.polynomial_degree + 1) - 1
    return first_element:last_element
end

function get_local_knot_vector(bspline::BSplineSpace, basis_idx::Int)
    knot_vector = bspline.knot_vector
    deg = get_polynomial_degree(bspline)

    knot_cum_sum = cumsum(knot_vector.multiplicity)

    first_breakpoint_idx = convert_knot_to_breakpoint_idx(knot_vector, basis_idx)
    last_breakpoint_idx = convert_knot_to_breakpoint_idx(knot_vector, basis_idx + deg + 1)

    first_knot_mult = knot_cum_sum[first_breakpoint_idx] - basis_idx + 1
    last_knot_mult = basis_idx + deg + 1 - knot_cum_sum[last_breakpoint_idx - 1]

    breakpoints = get_patch(bspline).breakpoints[first_breakpoint_idx:last_breakpoint_idx]
    multiplicity = vcat(first_knot_mult, get_multiplicity_vector(bspline)[first_breakpoint_idx+1:last_breakpoint_idx-1], last_knot_mult)

    return KnotVector(Mesh.Patch1D(breakpoints), deg, multiplicity)
end

function get_max_local_dim(bspline::BSplineSpace)
    return bspline.knot_vector.polynomial_degree + 1
end

function get_base_control_points(bspline::BSplineSpace)
    return get_greville_points(bspline.knot_vector)
end

"""
    get_dof_partition(bspline::BSplineSpace)

Get the partition of degrees of freedom for the B-spline space

# Arguments
- `bspline::BSplineSpace`: The B-spline space.

# Returns
The partition of degrees of freedom.
"""
function get_dof_partition(bspline::BSplineSpace)
    return bspline.dof_partition
end

"""
    assemble_global_extraction_matrix(bsplines::BSplineSpace)

Loops over all elements and assembles the global extraction matrix for the B-spline space. The extraction matrix is a sparse matrix that maps the local basis functions to the global basis functions.

# Arguments
- `bspline::BSplineSpace`: The B-spline space.

# Returns
- `::Array{Float64,2}`: Global extraction matrix.
"""
function assemble_global_extraction_matrix(bspline::BSplineSpace)
    # Number of global basis functions
    num_global_basis = get_num_basis(bspline)
    # Number of elements
    nel = get_num_elements(bspline)
    # Number of local basis functions
    num_local_basis = repeat([get_polynomial_degree(bspline.polynomials) + 1], nel, 1)
    num_local_basis_offset = cumsum([0; num_local_basis])
    # Initialize the global extraction matrix
    global_extraction_matrix = zeros(Float64, num_local_basis_offset[end], num_global_basis)

    # Loop over all elements
    for el_id ∈ 1:nel
        # get extraction on this element
        extraction_coefficients, global_basis_indices = get_extraction(bspline, el_id)
        # get local basis indices
        local_basis_indices = num_local_basis_offset[el_id]+ 1:num_local_basis_offset[el_id + 1]

        # Assemble the global extraction matrix
        global_extraction_matrix[local_basis_indices, global_basis_indices] = extraction_coefficients
    end

    return SparseArrays.sparse(global_extraction_matrix)
end

"""
    get_derivative_space(bspline::BSplineSpace{F}) where {F <: AbstractCanonicalSpace}

Returns the derivative space of the B-spline space.

# Arguments
- `bspline::BSplineSpace{F}`: The B-spline space.

# Returns
- `::BSplineSpace{F}`: The derivative space.
"""
function get_derivative_space(bspline::BSplineSpace{F}) where {F <: AbstractCanonicalSpace}
    # polynomial degree of derivative space
    p = get_polynomial_degree(bspline)
    if F <: AbstractECTSpaces
        dpolynomials = get_derivative_space(F)
    else
        dpolynomials = F(p-1)
    end

    # modified left and right dof-partitioning
    dof_partition = get_dof_partition(bspline)
    n_left = max(0, length(dof_partition[1][1])-1)
    n_right = max(0, length(dof_partition[1][3])-1)
    
    # regularity of derivative space
    dregularity = (p-1) .- get_multiplicity_vector(bspline)
    for i in eachindex(dregularity)
        if dregularity[i] < -1
            dregularity[i] = -1
        end
    end

    return BSplineSpace(get_patch(bspline), dpolynomials, dregularity, n_left, n_right)
end

# Methods for ease of function space creation

"""
    create_bspline_space(starting_point::Float64, box_size::Float64, num_elements::Int, degree::Int, regularity::Int)

Create a 1D B-spline space based on the provided starting point, box size, number of elements, degree, and regularity. It constructs a uniform distribution of breakpoints and uses them to define a `Mesh.Patch1D` object. The function then creates a regularity vector, with the first and last entries set to -1 to ensure an open knot vector, and returns the corresponding `BSplineSpace`.

# Arguments
- `starting_point::Float64`: The coordinate of the starting point of the B-spline space.
- `box_size::Float64`: The size of the space along the single dimension.
- `num_elements::Int`: The number of elements along the dimension.
- `degree::Int`: The degree of the B-spline basis functions.
- `regularity::Int`: The regularity (smoothness) between consecutive B-spline basis functions, typically between 0 and `degree - 1`.

# Returns
- `::BSplineSpace`: The generated B-spline space.

# Examples
```julia
# Create a 1D B-spline space
start_1d = 0.0
size_1d = 10.0
elements_1d = 100
degree = 3
regularity = 2
bspline_space_1d = create_bspline_space(start_1d, size_1d, elements_1d, degree, regularity)
```
"""
function create_bspline_space(starting_point::Float64, box_size::Float64, num_elements::Int, degree::Int, regularity::Int)

    box_size > 0.0 ? nothing : throw(ArgumentError("The argumente 'box_size' must be greater than 0."))

    breakpoints = collect(LinRange(starting_point, starting_point+box_size, num_elements+1))
    patch = Mesh.Patch1D(breakpoints)

    regularity_vector = fill(regularity, (num_elements+1,))
    regularity_vector[1] = regularity_vector[end] = -1 # Open knot vector

    return BSplineSpace(patch, degree, regularity_vector)
end

"""
    create_bspline_space(starting_point::NTuple{1, Float64}, box_size::NTuple{1, Float64}, num_elements::NTuple{1, Int}, degree::NTuple{1, Int}, regularity::NTuple{1, Int})

Create a 1D B-spline space based on the provided starting point, box size, number of elements, degree, and regularity. It constructs a uniform distribution of breakpoints and uses them to define a `Mesh.Patch1D` object. The function then creates a regularity vector, with the first and last entries set to -1 to ensure an open knot vector, and returns the corresponding `BSplineSpace`.

# Arguments
- `starting_point::NTuple{1, Float64}`: The coordinate of the starting point of the B-spline space.
- `box_size::NTuple{1, Float64}`: The size of the space along the single dimension.
- `num_elements::NTuple{1, Int}`: The number of elements along the dimension.
- `degree::NTuple{1, Int}`: The degree of the B-spline basis functions.
- `regularity::NTuple{1, Int}`: The regularity (smoothness) between consecutive B-spline basis functions, typically between 0 and `degree - 1`.

# Returns
- `::BSplineSpace`: The generated B-spline space.

# Examples
```julia
# Create a 1D B-spline space
start_1d = 0.0
size_1d = 10.0
elements_1d = 100
degree = 3
regularity = 2
bspline_space_1d = create_bspline_space(start_1d, size_1d, elements_1d, degree, regularity)
```
"""
function create_bspline_space(starting_point::NTuple{1, Float64}, box_size::NTuple{1, Float64}, num_elements::NTuple{1, Int}, degree::NTuple{1, Int}, regularity::NTuple{1, Int})
    return create_bspline_space(starting_point[1], box_size[1], num_elements[1], degree[1], regularity[1])
end