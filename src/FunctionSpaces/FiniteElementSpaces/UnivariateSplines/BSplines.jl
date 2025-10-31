"""
    BSplineSpace{F, TE, TI, TJ} <: AbstractFESpace{1, 1, 1}

Structure containing information about a univariate B-Spline function space defined on
`patch_1d::Mesh.Patch1D`, with given `polynomial_degree` and `regularity` per breakpoint.
Note that while the section spaces on each element are the same, they don't necessarily have
to be polynomials; they are just named `polynomials` for convention.

# Fields
- `knot_vector::KnotVector`: 1-dimensional knot vector.
- `extraction_op::ExtractionOperator`: Stores extraction coefficients and basis indices.
- `polynomials::F`: local section space F, named `polynomials` just for convention.
- `dof_partition::Vector{Vector{Vector{Int}}}`: Indices of boundary degrees of freedom.
"""
struct BSplineSpace{F, TE, TI, TJ} <: AbstractFESpace{1, 1, 1}
    knot_vector::KnotVector
    extraction_op::ExtractionOperator{1, TE, TI, TJ}
    polynomials::F
    dof_partition::Vector{Vector{Vector{Int}}}

    function BSplineSpace(
        patch_1d::Mesh.Patch1D,
        polynomials::F,
        regularity::Vector{Int},
        n_dofs_left::Int=1,
        n_dofs_right::Int=1,
    ) where {F <: AbstractCanonicalSpace}
        polynomial_degree = get_polynomial_degree(polynomials)

        if polynomial_degree < 0
            throw(ArgumentError("""\
                The polynomial degree must be greater than or equal to 0, but is \
                $polynomial_degree.\
                """))
        end

        num_breakpoints = size(patch_1d) + 1
        if num_breakpoints != length(regularity)
            throw(ArgumentError("""\
                The number of regularity conditions should be equal to the number of \
                breakpoints, but there are $(num_breakpoints) breakpoints and \
                $(length(regularity)) regularity conditions.\
                """))
        end

        for i in eachindex(regularity)
            if polynomial_degree <= regularity[i]
                throw(
                    ArgumentError("""\
                  The polynomial degree must be greater than the regularity, but the \
                  polynomial degree is $polynomial_degree and the regularity at index $i \
                  is $(regularity[i]).\
                  """)
                )
            end
        end

        for i in eachindex(regularity)
            if regularity[i] < -1
                throw(ArgumentError("""\
                    The minimum regularity is -1 (element-wise discontinuous), but the \
                    regularity at index $i is $(regularity[i]).\
                    """))
            end
        end

        if F <: AbstractLagrangePolynomials
            if maximum(regularity) > 0
                throw(
                    ArgumentError("""\
                  The regularity conditions for Lagrange polynomials must be -1 \
                  (discontinuous) or 0 (C^0 continuous). You have regularity conditions \
                  $(regularity), which has maximum $(maximum(regularity)).\
                  """)
                )
            end
        end

        knot_vector = create_knot_vector(
            patch_1d, polynomial_degree, regularity, "regularity"
        )
        extraction_op = extract_bspline_to_section_space(knot_vector, polynomials)
        bspline_dim = get_num_basis(extraction_op)

        dof_partition = Vector{Vector{Vector{Int}}}(undef, 1)
        dof_partition[1] = Vector{Vector{Int}}(undef, 3)
        # First, store the left dofs ...
        dof_partition[1][1] = collect(1:n_dofs_left)
        # ... then the interior dofs ...
        dof_partition[1][2] = collect((n_dofs_left + 1):(bspline_dim - n_dofs_right))
        # ... and then finally the right dofs.
        dof_partition[1][3] = collect((bspline_dim - n_dofs_right + 1):bspline_dim)

        return new{F, get_EIJ_types(extraction_op)...}(
            knot_vector, extraction_op, polynomials, dof_partition
        )
    end
end

# Helper functions with classical choices for defaults.
function BSplineSpace(
    patch_1d::Mesh.Patch1D, polynomial_degree::Int, regularity::Vector{Int}
)
    return BSplineSpace(patch_1d, Bernstein(polynomial_degree), regularity)
end
function BSplineSpace(
    patch_1d::Mesh.Patch1D, polynomials::AbstractCanonicalSpace, regularity::Int
)
    # Open knot vector (-1 regularity at the endpoints), given internal regularity.
    regularity = [-1; repeat([regularity], size(patch_1d) - 1); -1]
    return BSplineSpace(patch_1d, polynomials, regularity)
end
function BSplineSpace(patch_1d::Mesh.Patch1D, polynomial_degree::Int, regularity::Int)
    regularity = [-1; repeat([regularity], size(patch_1d) - 1); -1]
    return BSplineSpace(patch_1d, Bernstein(polynomial_degree), regularity)
end

"""
    get_knot_vector(space::BSplineSpace)

Returns the knot vector of the B-spline space `space`.

# Arguments
- `space::BSplineSpace`: The B-spline space.

# Returns
- `::KnotVector`: The knot vector of the B-spline space.
"""
function get_knot_vector(space::BSplineSpace)
    return space.knot_vector
end

"""
    get_polynomials(space::BSplineSpace)

Returns the reference Bernstein polynomials of `space`.

# Arguments
- `space::BSplineSpace`: A univariate B-Spline function space.

# Returns
- `::Bernstein`: Bernstein polynomials.
"""
function get_polynomials(space::BSplineSpace)
    return space.polynomials
end

function get_local_basis(
    space::BSplineSpace,
    element_id::Int,
    xi::Points.AbstractPoints{1},
    nderivatives::Int,
    component_id::Int=1,
)
    # The output of this function must correspond to the general evaluate function, so the
    # output must be a vector{vector{vector{Matrix{Float64}}}}. The output of the evaluate
    # on polynomials is a vector{vector{Matrix{Float64}}}, so we need to add an extra layer
    # of vectors to the output, corresponding to the component.
    section_space_eval = evaluate(get_polynomials(space), xi, nderivatives)
    ext_eval = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
    for i in 1:(nderivatives + 1)
        # The section spaces, which are CanonicalSpaces, are always 1D, so one derivative
        # per derivative order.
        ext_eval[i] = Vector{Vector{Matrix{Float64}}}(undef, 1)
        ext_eval[i][1] = [section_space_eval[i][1]]
    end

    return ext_eval
end

# Note that `elem_id` is an optional dummy argument for uniformity with other spaces (dummy
# because the degree is the same for all elements for B-splines).
function get_polynomial_degree(space::BSplineSpace, elem_id::Int=0)
    return get_polynomial_degree(get_polynomials(space))
end

"""
    get_patch(space::BSplineSpace)

Returns the patch of the univariate function space `space`.

# Arguments
- `space::BSplineSpace`: The B-Spline function space.

# Returns
- `::Mesh.Patch1D`: The patch of the B-Spline space.
"""
function get_patch(space::BSplineSpace)
    return get_knot_vector(space).patch_1d
end

"""
    get_multiplicity_vector(space::BSplineSpace)

Returns the multiplicities of the knot vector associated with the univariate function space
`space`.

# Arguments
- `space::BSplineSpace`: The B-Spline function space.

# Returns
- `::Vector{Int}`: The multiplicity of the knot vector associated with the B-Spline space.
"""
function get_multiplicity_vector(space::BSplineSpace)
    return get_knot_vector(space).multiplicity
end

function get_num_elements(space::BSplineSpace)
    return size(get_knot_vector(space).patch_1d)
end

"""
    get_element_measure(space::BSplineSpace, element_id::Int)

Returns the size of the element specified by `element_id`.

# Arguments
- `space::BSplineSpace`: The B-Spline function space.
- `element_id::Int`: The id of the element.

# Returns
- `::Float64`: The size of the element.
"""
function get_element_measure(space::BSplineSpace, element_id::Int)
    return get_element_measure(get_knot_vector(space), element_id)
end

function get_element_lengths(space::BSplineSpace, element_id::Int)
    return get_element_measure(get_knot_vector(space), element_id)
end

"""
    get_element_vertices(space::BSplineSpace, element_id::Int)

Returns the vertices of the element specified by `element_id`.

# Arguments
- `space::BSplineSpace`: The B-Spline function space.
- `element_id::Int`: The id of the element.

# Returns
- `::NTuple{1, Vector{Float64}`: The vertices of the element.
"""
function get_element_vertices(space::BSplineSpace, element_id::Int)
    return Mesh.get_element_vertices(get_patch(space), element_id)
end

"""
    get_support(space::BSplineSpace, basis_id::Int)

Returns the elements where the B-spline given by `basis_id` is supported.

# Arguments
- `space::BSplineSpace`: The B-Spline function space.
- `basis_id::Int`: The id of the basis function.

# Returns
- `::Vector{Int}`: The support of the basis function.
"""
function get_support(space::BSplineSpace, basis_id::Int)
    first_element = convert_knot_to_breakpoint_idx(get_knot_vector(space), basis_id)
    last_element =
        convert_knot_to_breakpoint_idx(
            get_knot_vector(space), basis_id + get_knot_vector(space).polynomial_degree + 1
        ) - 1
    return collect(first_element:last_element)
end

function get_local_knot_vector(space::BSplineSpace, basis_idx::Int)
    knot_vector = get_knot_vector(space)
    deg = get_polynomial_degree(space)

    knot_cum_sum = cumsum(knot_vector.multiplicity)

    first_breakpoint_idx = convert_knot_to_breakpoint_idx(knot_vector, basis_idx)
    last_breakpoint_idx = convert_knot_to_breakpoint_idx(knot_vector, basis_idx + deg + 1)

    first_knot_mult = knot_cum_sum[first_breakpoint_idx] - basis_idx + 1
    last_knot_mult = basis_idx + deg + 1 - knot_cum_sum[last_breakpoint_idx - 1]

    breakpoints = get_patch(space).breakpoints[first_breakpoint_idx:last_breakpoint_idx]
    multiplicity = vcat(
        first_knot_mult,
        get_multiplicity_vector(space)[(first_breakpoint_idx + 1):(last_breakpoint_idx - 1)],
        last_knot_mult,
    )

    return KnotVector(Mesh.Patch1D(breakpoints), deg, multiplicity)
end

function get_max_local_dim(space::BSplineSpace)
    return get_knot_vector(space).polynomial_degree + 1
end

function get_greville_points(space::BSplineSpace)
    return get_greville_points(get_knot_vector(space))
end

function assemble_global_extraction_matrix(space::BSplineSpace)
    # Number of global basis functions
    num_global_basis = get_num_basis(space)
    # Number of elements
    nel = get_num_elements(space)
    # Number of local basis functions
    num_local_basis = (get_polynomial_degree(get_polynomials(space)) + 1) .* ones(Int, nel)
    num_local_basis_offset = cumsum([0; num_local_basis])
    # Initialize the global extraction matrix
    global_extraction_matrix = zeros(Float64, num_local_basis_offset[end], num_global_basis)

    # Loop over all elements
    for el_id in 1:nel
        # get extraction on this element
        extraction_coefficients = get_extraction_coefficients(space, el_id)
        global_basis_indices = get_basis_indices(space, el_id)
        # get local basis indices
        local_basis_indices =
            (num_local_basis_offset[el_id] + 1):num_local_basis_offset[el_id + 1]

        # Assemble the global extraction matrix
        global_extraction_matrix[local_basis_indices, global_basis_indices] =
            extraction_coefficients
    end

    return SparseArrays.sparse(global_extraction_matrix)
end

"""
    get_derivative_space(space::BSplineSpace)

Returns the derivative space of the B-spline space.

# Arguments
- `space::BSplineSpace`: The B-spline space.

# Returns
- `::BSplineSpace`: The derivative space.
"""
function get_derivative_space(space::BSplineSpace)
    # polynomial degree of derivative space
    p = get_polynomial_degree(space)
    dpolynomials = get_derivative_space(get_polynomials(space))

    # modified left and right dof-partitioning
    dof_partition = get_dof_partition(space)
    n_left = max(0, length(dof_partition[1][1]) - 1)
    n_right = max(0, length(dof_partition[1][3]) - 1)

    # regularity of derivative space
    dregularity = (p - 1) .- get_multiplicity_vector(space)
    for i in eachindex(dregularity)
        if dregularity[i] < -1
            dregularity[i] = -1
        end
    end

    return BSplineSpace(get_patch(space), dpolynomials, dregularity, n_left, n_right)
end
