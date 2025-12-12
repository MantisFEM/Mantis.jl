############################################################################################
#                                       Subdivision                                        #
############################################################################################

"""
    subdivide_multiplicity_vector(
        parent_multiplicity::Vector{Int}, num_subdivisions::Int, child_multiplicity::Int
    )

Subdivides `parent_multiplicity` by uniformly subdiving each element `num_subdivisions` times.
The parent multiplicities are preserved in the `child_multiplicity_vector`, and newly
inserted ones are given multiplicity `child_multiplicity`.

# Arguments
- `parent_multiplicity::Vector{Int}`: parent multiplicity vector.
- `num_subdivisions::Int`: Number of times each element is subdivided.
- `child_multiplicity::Int`: Multiplicity of each new knot.

# Returns
- `child_multiplicity_vector::Vector{Int}`: child multiplicity vector.
"""
function subdivide_multiplicity_vector(
    parent_multiplicity::Vector{Int}, num_subdivisions::Int, child_multiplicity::Int=1
)
    mult_length = 1 + length(parent_multiplicity) * (num_subdivisions) - num_subdivisions

    child_multiplicity_vector = fill(child_multiplicity, mult_length)

    parent_idx = 1
    for k in eachindex(child_multiplicity_vector)
        if (k - 1) % num_subdivisions + 1 == 1
            if child_multiplicity_vector[k] < parent_multiplicity[parent_idx]
                child_multiplicity_vector[k] = parent_multiplicity[parent_idx]
            end

            parent_idx += 1
        end
    end

    return child_multiplicity_vector
end

"""
    subdivide_knot_vector(
        parent_knot_vector::KnotVector, num_subdivisions::Int, child_multiplicity::Int
    )

Subdivides `parent_knot_vector` by uniformly subdiving each element `num_subdivisions` times.
The parent multiplicities are preserved in the `child_multiplicity_vector`, and newly
inserted ones are given multiplicity `child_multiplicity`.

# Arguments
- `parent_knot_vector::KnotVector`: parent knot vector.
- `num_subdivisions::Int`: Number of times each element is subdivided.
- `child_multiplicity::Int`: Multiplicity of each new knot.

# Returns
- `::KnotVector`: child knot vector.
"""
function subdivide_knot_vector(
    parent_knot_vector::KnotVector, num_subdivisions::Int, child_multiplicity::Int
)
    child_geometry = subdivide_geometry(get_geometry(parent_knot_vector), num_subdivisions)
    child_multiplicity_vector = subdivide_multiplicity_vector(
        get_multiplicity(parent_knot_vector), num_subdivisions, child_multiplicity
    )
    p = get_polynomial_degree(parent_knot_vector)

    return KnotVector(child_geometry, p, child_multiplicity_vector)
end

"""
    subdivide_space(
        parent_bspline::BSplineSpace, num_subdivisions::Int, child_multiplicity::Int
    )

Subdivides `parent_bspline` by uniformly subdiving each element `num_subdivisions` times.
The parent multiplicities are preserved in the final multiplicity vector, and newly inserted
ones are given multiplicity `child_multiplicity`.

# Arguments
- `parent_bspline::BSplineSpace`: parent B-spline.
- `num_subdivisions::Int`: Number of times each element is subdivided.
- `child_multiplicity::Int`: Multiplicity of each new knot.

# Returns
- `::BSplineSpace`: refined B-spline space.
"""
function subdivide_space(
    parent_bspline::BSplineSpace, num_subdivisions::Int, child_multiplicity::Int=1
)
    child_knot_vector = subdivide_knot_vector(
        get_knot_vector(parent_bspline), num_subdivisions, child_multiplicity
    )
    child_parametric_geometry = get_geometry(child_knot_vector)
    child_geometry = subdivide_geometry(get_geometry(parent_bspline), num_subdivisions)
    child_polynomials = get_child_canonical_space(
        get_polynomials(parent_bspline), num_subdivisions
    )
    dof_partition = get_dof_partition(parent_bspline)
    n_dofs_left = length(dof_partition[1][1])
    n_dofs_right = length(dof_partition[1][3])
    p = get_polynomial_degree(parent_bspline)

    return BSplineSpace(
        child_geometry,
        child_parametric_geometry,
        child_polynomials,
        p .- get_multiplicity(child_knot_vector),
        n_dofs_left,
        n_dofs_right,
    )
end

############################################################################################
#                                   Build two-scale data                                   #
############################################################################################
#
function build_two_scale_operator(
	parent_bspline::BSplineSpace, num_subdivisions::Tuple{Int}
)
	return build_two_scale_operator(parent_bspline, num_subdivisions[1])
end

"""
    build_two_scale_operator(
        parent_bspline::BSplineSpace, num_subdivisions::Int, child_multiplicity::Int
    )
Algorithm for the coefficients of a change of B-spline representation for knot insertion
of multiple knots, recursively using `single_knot_insertion_oslo()`. The parent knot vector
is `parent_bspline.knot_vector` and the inserted knots are given by `num_subdivisions`,
meaning `num_subdivisions-1` uniformly spaced knots are inserted, between parent breakpoints,
with multiplicity 1.

For more information, see [Dangella2018](@cite).

# Arguments
- `parent_bspline::BSplineSpace`: parent B-spline.
- `num_subdivisions::Int`: Number of times each element is subdivided.
- `child_multiplicity::Int`: Multiplicity of each new knot in refined knot vector.\

# Returns
- `::FiniteElementSpaces.TwoScaleOperator, child_bspline::BSplineSpace`: Tuple with a
    twoscale_operator and child B-spline space.
"""
function build_two_scale_operator(
    parent_bspline::BSplineSpace, num_subdivisions::Int, child_multiplicity::Int=1
)
    if num_subdivisions <= 0
        throw(ArgumentError("Number of subdivions must be greater than 0.
                            num_subdivisions=$num_subdivisions was given."))
    end

    child_bspline = subdivide_space(parent_bspline, num_subdivisions, child_multiplicity)

    return build_two_scale_operator(parent_bspline, child_bspline, num_subdivisions)
end

"""
    build_two_scale_operator(
        parent_bspline::BSplineSpace{F}, child_bspline::BSplineSpace{F}, num_subdivisions::Int
    ) where {F <: AbstractCanonicalSpace}

Algorithm for the coefficients of a change of B-spline representation for knot insertion
of multiple knots, recursively using `single_knot_insertion_oslo()`. The parent knot vector
is `parent_bspline.knot_vector` and the inserted knots are given by
`child_bspline.knot_vector`.

For more information, see [Dangella2018](@cite).

# Arguments
- `parent_bspline::BSplineSpace`: parent B-spline.
- `child_bspline::BSplineSpace`: child B-spline, with extra knots.
- `num_subdivisions::Int`: Number of times each element is subdivided.

# Returns
- `::FiniteElementSpaces.TwoScaleOperator, child_bspline::BSplineSpace`: Tuple with a
    twoscale_operator and child B-spline space.
"""
function build_two_scale_operator(
    parent_bspline::BSplineSpace{F}, child_bspline::BSplineSpace{F}, num_subdivisions::Int
) where {F <: AbstractCanonicalSpace}
    if F <: Bernstein
        gm = build_two_scale_matrix(
            get_knot_vector(parent_bspline), get_knot_vector(child_bspline)
        )
    else
        # build the element subdivision matrix
        el_subdivision_mat = build_two_scale_matrix(
            parent_bspline.polynomials, num_subdivisions
        )
        # assemble the global extraction operators for the parent and child spaces
        parent_extraction_mat = assemble_global_extraction_matrix(parent_bspline)
        child_extraction_mat = assemble_global_extraction_matrix(child_bspline)
        # concatenate the two_scale_operator subdivision matrices in a block diagonal format
        discont_subdivision_mat = SparseArrays.blockdiag(
            [el_subdivision_mat for i in 1:get_num_elements(parent_bspline)]...
        )
        # compute the two-scale matrix by solving a least-squares problem
        gm = SparseArrays.sparse(
            child_extraction_mat \ Array(discont_subdivision_mat * parent_extraction_mat)
        )
        SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, gm)
    end

    parent_to_child_elements = get_parent_to_children_elements(
        parent_bspline, num_subdivisions
    )
    child_to_parent_elements = get_child_to_parent_elements(child_bspline, num_subdivisions)

    return TwoScaleOperator(
        parent_bspline,
        child_bspline,
        gm,
        parent_to_child_elements,
        child_to_parent_elements,
    ),
    child_bspline
end

"""
    build_two_scale_matrix(parent_knot_vector::KnotVector, child_knot_vector::KnotVector)

Algorithm for the coefficients of a change of B-spline representation for knot insertion
of multiple knots, recursively using `single_knot_insertion_oslo()`. The parent knot vector
is `parent_knot_vector` and the inserted knots are given by `child_knot_vector`.

For more information, see [Dangella2018](@cite).

# Arguments
- `parent_knot_vector::KnotVector`: parent knot vector.
- `child_knot_vector::KnotVector`: child knot vector, with the extra knots.

# Returns
- `global_extraction_matrix`: Global subdivision matrix
"""
function build_two_scale_matrix(
    parent_knot_vector::KnotVector, child_knot_vector::KnotVector
)
    m = get_knot_vector_length(child_knot_vector)
    nel = get_num_elements(child_knot_vector)
    p = get_polynomial_degree(parent_knot_vector)
    nchild = m - p - 1

    gm_values = zeros(Float64, nchild * (p + 1))
    gm_rows = zeros(Int, nchild * (p + 1))
    gm_columns = similar(gm_rows)
    sparse_idx = zeros(Int, p + 1)

    cf = p + 1
    rf = 1
    e = 1

    local_subdiv_matrix = [Matrix{Float64}(LinearAlgebra.I, p + 1, p + 1) for _ in 1:nel]

    offs = 0

    while rf <= nchild
        mult = get_knot_multiplicity(child_knot_vector, rf)

        lastcf = cf
        while get_knot_value(parent_knot_vector, cf + 1) <=
              get_knot_value(child_knot_vector, rf)
            cf += 1
        end

        if e > 1
            offs = cf - lastcf
            local_subdiv_matrix[e][1:(p + 1 - offs), 1:(p + 1 - mult)] .= local_subdiv_matrix[e - 1][
                (1 + offs):(p + 1), (1 + mult):(p + 1)
            ]
        end

        for t in (p + 2 - mult):(p + 1)
            sparse_idx .= ((rf - 1) * (p + 1) + 1):(rf * (p + 1))
            gm_columns[sparse_idx] .= (cf - p):cf
            gm_rows[sparse_idx] .= rf

            local_subdiv_matrix[e][:, t] = single_knot_insertion_oslo(
                parent_knot_vector, child_knot_vector, cf, rf
            )
            gm_values[sparse_idx] .= local_subdiv_matrix[e][:, t]

            rf += 1
        end

        e += 1
    end

    global_extraction_matrix = SparseArrays.dropzeros(
        SparseArrays.sparse(gm_rows, gm_columns, gm_values, rf - 1, cf)
    )

    return global_extraction_matrix
end

"""
    single_knot_insertion_oslo(
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
function single_knot_insertion_oslo(
    parent_knot_vector::KnotVector, child_knot_vector::KnotVector, cf::Int, rf::Int
)
    b = [1.0]
    p = get_polynomial_degree(parent_knot_vector)
    for k in 1:p
        t1 = get_knot_value.((parent_knot_vector,), (cf + 1 - k):cf)
        t2 = get_knot_value.((parent_knot_vector,), (cf + 1):(cf + k))
        x = get_knot_value(child_knot_vector, rf + k)
        w = (x .- t1) ./ (t2 .- t1)
        b = push!((1 .- w) .* b, 0) .+ pushfirst!(w .* b, 0)
    end

    return b
end

############################################################################################
#                                Parent-to-child relations                                 #
############################################################################################

function get_parent_to_children_elements(::BSplineSpace, num_subdivisions::Int)
    return parent -> get_element_children(parent, num_subdivisions)
end

function get_child_to_parent_elements(::BSplineSpace, num_subdivisions::Int)
    return child -> get_element_parent(child, num_subdivisions)
end
