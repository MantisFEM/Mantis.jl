"""
    BSplineSpace{F, G, GP, TM, TE, TI, TJ, D} <: AbstractFESpace{1, 1, 1}

Univariate (general) B-Spline function space defined on a `knot_vector::KnotVector`, with
given `polynomial_degree` and `regularity` per breakpoint.

Note that while the section spaces on each element are the same, they don't necessarily have
to be polynomials; they are just named `polynomials` for convention.

# Fields
- `geometry::G`: Physical geometry to which the B-Spline space is mapped.
- `knot_vector::KnotVector`: 1-dimensional knot vector defining the parametric geometry. See
    [`KnotVector`](@ref) for more details.
- `extraction_op::ExtractionOperator`: Stores extraction coefficients and basis indices.
- `polynomials::F`: local section space F, named `polynomials` just for convention. Can be
    any [`AbstractCanonicalSpace`](@ref).
- `dof_partition::D`: Partitioning of the degrees of freedom into boundary and interior
    dofs. The type will be similar to Vector{Vector{Vector{Int}}}, where the outer Vector
    will have length 1 (its for patches), the middle Vector will be of length 3 (1: left
    boundary, 2 interior, 3 right boundary), and the inner Vector will contain the global
    basis indices.
"""
struct BSplineSpace{F, G, GP, TM, TE, TI, TJ, D} <: AbstractFESpace{1, 1, 1}
    geometry::G
    knot_vector::KnotVector{GP, TM}
    polynomials::F
    extraction_op::ExtractionOperator{1, TE, TI, TJ}
    dof_partition::D

    function BSplineSpace(
        geometry::G,
        parametric_geometry::GP,
        polynomials::F,
        regularity::Vector{Int},
        n_dofs_left::Int=1,
        n_dofs_right::Int=1,
    ) where {
        image_dim,
        F <: AbstractCanonicalSpace,
        G <: Geometry.AbstractGeometry{1, image_dim, 1},
        GP <: Geometry.CartesianGeometry{1, image_dim, 1},
    }
        polynomial_degree = get_polynomial_degree(polynomials)
        if polynomial_degree < 0
            throw(
                ArgumentError(
                    LazyString(
                        "The polynomial degree must be greater than or equal to 0, but is ",
                        polynomial_degree,
                        ".",
                    ),
                ),
            )
        end

        breakpoints = Geometry.get_breakpoints_per_dim(parametric_geometry)
        num_breakpoints = length(breakpoints)
        if num_breakpoints != length(regularity)
            throw(
                ArgumentError(
                    LazyString(
                        "The number of regularity conditions should be equal to the number",
                        " of breakpoints, but there are ",
                        num_breakpoints,
                        " breakpoints and ",
                        length(regularity),
                        " regularity conditions.",
                    ),
                ),
            )
        end

        for i in eachindex(regularity)
            if polynomial_degree <= regularity[i]
                throw(
                    ArgumentError(
                        LazyString(
                            "The polynomial degree must be greater than the regularity, ",
                            "but the polynomial degree is ",
                            polynomial_degree,
                            " and the regularity at index ",
                            i,
                            " is ",
                            regularity[i],
                            ".",
                        ),
                    ),
                )
            end

            if regularity[i] < -1
                throw(
                    ArgumentError(
                        LazyString(
                            "The minimum regularity is -1 (element-wise discontinuous), ",
                            "but the regularity at index ",
                            i,
                            " is ",
                            regularity[i],
                            ".",
                        ),
                    ),
                )
            end

            if F <: AbstractLagrangePolynomials && regularity[i] > 0
                throw(
                    ArgumentError(
                        LazyString(
                            "The regularity conditions for Lagrange polynomials must be -1",
                            "(discontinuous) or 0 (C^0 continuous), but you have ",
                            "regularity condition ",
                            regularity[i],
                            " at index ",
                            i,
                            ".",
                        ),
                    ),
                )
            end
        end

        knot_vector = create_knot_vector(
            parametric_geometry, polynomial_degree, regularity, "regularity"
        )
        extraction_op = extract_bspline_to_section_space(knot_vector, polynomials)
        bspline_dim = get_num_basis(extraction_op)

        # A BSplineSpace is always single patch, so the outer most vector has no additional
        # entries.
        dof_partition = [[
            collect(1:n_dofs_left),  # Left dofs
            collect((n_dofs_left + 1):(bspline_dim - n_dofs_right)),  # Interior dofs
            collect((bspline_dim - n_dofs_right + 1):bspline_dim),  # Right dofs
        ]]
        return new{
            F,
            G,
            get_knot_vector_types(knot_vector)...,
            get_EIJ_types(extraction_op)...,
            typeof(dof_partition),
        }(
            geometry, knot_vector, polynomials, extraction_op, dof_partition
        )
    end
end

# Constructors with classical choices for defaults.
# General constructor, given only 1 CartesianGeometry, which is used as both parametric and
# physical geometry.
function BSplineSpace(
    geometry::Geometry.CartesianGeometry{1, image_dim, 1},
    polynomials::AbstractCanonicalSpace,
    regularity::Vector{Int},
) where {image_dim}
    return BSplineSpace(geometry, geometry, polynomials, regularity)
end
function BSplineSpace(
    geometry::Geometry.CartesianGeometry{1, image_dim, 1},
    mapping::Geometry.AbstractMapping{1, image_dim},
    polynomials::AbstractCanonicalSpace,
    regularity::Vector{Int},
) where {image_dim}
    physical_geometry = Geometry.MappedGeometry(geometry, mapping)
    return BSplineSpace(physical_geometry, geometry, polynomials, regularity)
end
# Given polynomial degree and regularity vector, assume the polynomial is Bernstein.
function BSplineSpace(
    geometry::Geometry.CartesianGeometry{1, image_dim, 1},
    polynomial_degree::Int,
    regularity::Vector{Int},
) where {image_dim}
    return BSplineSpace(geometry, geometry, Bernstein(polynomial_degree), regularity)
end
# Given polynomial degree and regularity value, assume the polynomial is Bernstein and the
# regularity is the same everywhere except at the endpoints (open knot vector).
function BSplineSpace(
    geometry::Geometry.CartesianGeometry{1, image_dim, 1},
    polynomial_degree::Int,
    regularity::Int,
) where {image_dim}
    breakpoints = Geometry.get_breakpoints_per_dim(geometry)
    num_breakpoints = length(breakpoints)
    regularity = [-1; repeat([regularity], num_breakpoints - 2); -1]
    return BSplineSpace(geometry, geometry, Bernstein(polynomial_degree), regularity)
end

# Given a polynomial and regularity vector, assume the polynomial is Bernstein.
function BSplineSpace(
    geometry::Geometry.CartesianGeometry{1, image_dim, 1},
    polynomials::AbstractCanonicalSpace,
    regularity::Int,
) where {image_dim}
    # Open knot vector (-1 regularity at the endpoints), given internal regularity.
    breakpoints = Geometry.get_breakpoints_per_dim(geometry)
    num_breakpoints = length(breakpoints)
    regularity = [-1; repeat([regularity], num_breakpoints - 2); -1]
    return BSplineSpace(geometry, geometry, polynomials, regularity)
end

function get_parametric_geometry(space::BSplineSpace)
    return get_geometry(get_knot_vector(space))
end

"""
    get_knot_vector(space::BSplineSpace)

Returns the knot vector object of the B-spline space `space`.

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
    # output must be a vector{vector{vector{Matrix{eltype(xi)}}}}. The output of the
    # evaluate on polynomials is a vector{vector{Matrix{eltype(xi)}}}, so we need to add an
    # extra layer of vectors to the output, corresponding to the component.
    section_space_eval = evaluate(get_polynomials(space), xi, nderivatives)
    ext_eval = Vector{Vector{Vector{Matrix{eltype(xi)}}}}(undef, nderivatives + 1)
    for i in eachindex(section_space_eval, ext_eval)
        # The section spaces, which are CanonicalSpaces, are always 1D, so one derivative
        # per derivative order.
        ext_eval[i] = Vector{Vector{Matrix{eltype(xi)}}}(undef, 1)
        ext_eval[i][1] = [section_space_eval[i][1]]
    end
    # npoints = Points.get_num_points(xi)
    # p = get_polynomial_degree(space)
    # ext_eval = Vector{Vector{Vector{Matrix{eltype(xi)}}}}(undef, nderivatives + 1)
    # for j in 0:nderivatives
    #     ext_eval[j + 1] = [[zeros(eltype(xi), npoints, p + 1)]]
    # end

    # ext_eval = [[[zeros(eltype(xi), npoints, p + 1)]] for j in 0:nderivatives]

    # evaluate!(ext_eval, get_polynomials(space), xi)

    return ext_eval
end

# Note that `elem_id` is an optional dummy argument for uniformity with other spaces (dummy
# because the degree is the same for all elements for B-splines).
function get_polynomial_degree(space::BSplineSpace, elem_id::Int=0)
    return get_polynomial_degree(get_polynomials(space))
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
    return get_multiplicity(get_knot_vector(space))
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

function get_local_knot_vector(space::BSplineSpace, basis_id::Int)
    knot_vector = get_knot_vector(space)

    return get_local_knot_vector(knot_vector, basis_id)
end

function get_max_local_dim(space::BSplineSpace)
    return get_polynomial_degree(space) + 1
end

function get_greville_points(space::BSplineSpace)
    return get_greville_points(get_knot_vector(space))
end

function get_breakpoints(space::BSplineSpace)
    return get_breakpoints(get_knot_vector(space))
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

    return BSplineSpace(
        get_geometry(space),
        get_parametric_geometry(space),
        dpolynomials,
        dregularity,
        n_left,
        n_right,
    )
end
