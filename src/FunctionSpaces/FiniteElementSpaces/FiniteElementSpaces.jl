
"""
    get_component_spaces(space::AbstractFESpace)

Get the component spaces of the (multi-)component finite element space. For a single
component space, this function will return the original space. Note that, in the most
general case, the component spaces are not necessarily single-patch spaces, and the
component spaces may contribute to multiple components.

# Arguments
- `space::AbstractFESpace`: A (multi-)component finite element space.

# Returns
- `component_spaces::Tuple{AbstractFESpace}`: Tuple of single-component spaces.
"""
function get_component_spaces(space::AbstractFESpace)
    return space.component_spaces
end

function get_component_spaces(
    space::AbstractFESpace{manifold_dim, 1, num_patches}
) where {manifold_dim, num_patches}
    return (space,)
end

"""
    get_num_elements_per_patch(space::AbstractFESpace)

Get the number of elements per patch in the multi-patch finite element space. Assumes that
all component spaces have the same number of patches and elements (per patch).

# Arguments
- `space::AbstractFESpace`: The multi-patch space.

# Returns
- `::NTuple{num_patches, Int}`: The number of elements per patch.

# Exceptions
- Error "'get_num_elements_per_patch' not implemented": This error is thrown if no
    'get_num_elements_per_patch' method is defined for a single-component space (which can
    be a component of a multi-component space).
"""
function get_num_elements_per_patch(
    space::AbstractFESpace{manifold_dim, num_components, num_patches}
) where {manifold_dim, num_components, num_patches}
    return get_num_elements_per_patch(get_component_spaces(space)[1])
end

function get_num_elements_per_patch(
    space::AbstractFESpace{manifold_dim, 1, num_patches}
) where {manifold_dim, num_patches}
    return error("'get_num_elements_per_patch' not implemented for $(typeof(space))")
end

function get_num_elements_per_patch(
    space::AbstractFESpace{manifold_dim, 1, 1}
) where {manifold_dim}
    return (get_num_elements(space),)
end

"""
    get_patch_id(space::AbstractFESpace, element_id::Int)

Get the ID of the patch to which the specified global element belongs.

# Arguments
- `space::AbstractFESpace`: The multi-patch space.
- `element_id::Int`: The global element ID.

# Returns
- `::Int`: ID of the patch to which the element belongs.
"""
function get_patch_id(space::AbstractFESpace, element_id::Int)
    cumulative_elements = cumsum(get_num_elements_per_patch(space))
    for ci in eachindex(cumulative_elements)
        if element_id <= cumulative_elements[ci]
            return ci
        end
    end
    throw(
        ArgumentError(
            "Element ID $(element_id) exceeds the total number of elements in the space."
        ),
    )
end

function get_patch_id(
    space::AbstractFESpace{manifold_dim, num_components, 1}, element_id::Int
) where {manifold_dim, num_components}
    return 1
end

"""
    get_patch_and_local_element_id(space::AbstractFESpace, element_id::Int)

Get the constituent patch ID and local element ID for the specified global element ID.

# Arguments
- `space::AbstractFESpace`: The multi-patch space.
- `element_id::Int`: The global element ID.

# Returns
- `patch_id::Int`: The patch ID
- `local_element_id::Int`: The local element ID.
"""
function get_patch_and_local_element_id(space::AbstractFESpace, element_id::Int)
    patch_id = get_patch_id(space, element_id)

    elements_per_patch = get_num_elements_per_patch(space)
    local_element_id = element_id
    for i in eachindex(elements_per_patch)
        if i < patch_id
            local_element_id -= elements_per_patch[i]
        else
            break
        end
    end

    return patch_id, local_element_id
end

"""
    get_global_element_id(space::AbstractFESpace, patch_id::Int, local_element_id::Int)

Get the global element ID for the specified constituent patch ID and local element ID.

# Arguments
- `space::AbstractFESpace`: A (multi-)patch space.
- `patch_id::Int`: The constituent patch ID.
- `local_element_id::Int`: The local element ID.

# Returns
- `::Int`: The global element ID.
"""
function get_global_element_id(space::AbstractFESpace, patch_id::Int, local_element_id::Int)
    return sum(get_num_elements_per_patch(space)[begin:(patch_id - 1)]; init=0) +
           local_element_id
end

"""
    get_extraction_operator(space::AbstractFESpace)

Returns the extraction operator of the given `space`.

!!! warning "Not all AbstractFESpaces have an extraction operator."
    Having an explicitly defined extraction operator is not required for building a finite
    element space.

# Arguments
- `space::AbstractFESpace`: A finite element space.

# Returns
- `::ExtractionOperator`: The extraction operator.

# Exceptions
- Error "no field 'extraction_op'": This error is thrown if no explicitly defined
    extraction operator is present for this `space`.
"""
function get_extraction_operator(space::AbstractFESpace)
    return space.extraction_op
end

"""
    get_extraction(space::AbstractFESpace, element_id::Int, component_id::Int=1)

Returns the extraction coefficients and indices of the supported basis functions on the
given element `element_id` for the given component `component_id`.

# Arguments
- `space::AbstractFESpace`: A finite element space.
- `element_id::Int`: Identifier of the element.
- `component_id::Int=1`: The component ID. This is only relevant for multi-component
    spaces, and thus defaults to 1. While it is not needed for single-component spaces, it
    is still required for the function signature.

# Returns
- `::Matrix{Float64}`: The extraction coefficients.
- `::Vector{Int}`: The (global) basis functions supported on this element.

# Exceptions
- Error "no field 'extraction_op'": This error is thrown if no extraction operator is
    defined for `space`, and there is no specific `get_extraction` method for `space`
    either.
"""
function get_extraction(space::AbstractFESpace, element_id::Int, component_id::Int=1)
    return get_extraction(get_extraction_operator(space), element_id, component_id)
end

"""
    get_extraction_coefficients(
        space::AbstractFESpace, element_id::Int, component_id::Int=1
    )

Returns the extraction coefficients on the given element.

# Arguments
- `space::AbstractFESpace`: A finite element space.
- `element_id::Int`: Identifier of the element.
- `component_id::Int=1`: The component ID. This is only relevant for multi-component
    spaces, and thus defaults to 1. While it is not needed for single-component spaces, it
    is still required for the function signature.

# Returns
- `::Matrix{Float64}`: The extraction coefficients on the requested element.

# Exceptions
- Error "no field 'extraction_op'": This error is thrown if no extraction operator is
    defined for `space`, and there is no specific `get_extraction_coefficients` method for
    `space` either.
"""
function get_extraction_coefficients(
    space::AbstractFESpace, element_id::Int, component_id::Int=1
)
    return get_extraction_coefficients(
        get_extraction_operator(space), element_id, component_id
    )
end

"""
    get_basis_indices(space::AbstractFESpace, element_id::Int)

Get the global indices of the basis functions of `space` on element `element_id`.

# Arguments
- `space::AbstractFESpace`: Finite element space.
- `element_id::Int`: Identifier of the element.

# Returns
- `::TI`: Global indices of the basis functions supported on this element. The type `TI`
    is an vector-like object with integer type elements. See the documentation of `space` or
    [`Indices`](@ref) for more details.


# Exceptions
- Error "no field 'extraction_op'": This error is thrown if no extraction operator is
    defined for `space`, and there is no specific `get_basis_indices` method for `space`
    either.
"""
function get_basis_indices(space::AbstractFESpace, element_id::Int)
    return get_basis_indices(get_extraction_operator(space), element_id)
end

"""
    get_basis_permutation(space::AbstractFESpace, element_id::Int, component_id::Int=1)

Get the permutation' of the basis indices. This tells the `evaluate` function to which basis
on an element the evaluations correspond.

# Arguments
- `space::AbstractFESpace`: Finite element space.
- `element_id::Int`: Identifier of the element.
- `component_id::Int=1`: The component ID. This is only relevant for multi-component
    spaces, and thus defaults to 1. While it is not needed for single-component spaces, it
    is still required for the function signature.

# Returns
- `::TJ`: Permutations of indices of the basis functions supported on this element. The type
    `TJ` is a vector-like object with integer type elements. See the documentation of
    `space` or [`Indices`](@ref) for more details.


# Exceptions
- Error "no field 'extraction_op'": This error is thrown if no extraction operator is
    defined for `space`, and there is no specific `get_basis_indices` method for `space`
    either.
"""
function get_basis_permutation(space::AbstractFESpace, element_id::Int, component_id::Int=1)
    return get_basis_permutation(get_extraction_operator(space), element_id, component_id)
end

"""
    get_num_basis(space::AbstractFESpace)

Returns the number of basis functions of the finite element space `space`.

# Arguments
- `space::AbstractFESpace`: Finite element space.

# Returns
- `::Int`: Number of basis functions spanning the finite element space.

# Exceptions
- Error "no field 'extraction_op'": This error is thrown if no extraction operator is
    defined for `space`, and there is no specific `get_num_basis` method for `space`
    either.
"""
function get_num_basis(space::AbstractFESpace)
    return get_num_basis(get_extraction_operator(space))
end

"""
    get_num_basis(space::AbstractFESpace, element_id::Int)

Get the number of basis functions of the finite element space `space` supported on element
`element_id`.

# Arguments
- `space::AbstractFESpace`: Finite element space.
- `element_id::Int`: Indentifier of the element.

# Returns
- `::Int`: Number of basis functions supported on the given element.

# Exceptions
- Error "no field 'extraction_op'": This error is thrown if no extraction operator is
    defined for `space`, and there is no specific `get_num_basis` method for `space`
    either.
"""
function get_num_basis(space::AbstractFESpace, element_id::Int)
    return length(get_basis_indices(space, element_id))
end

"""
    get_dof_partition(space::AbstractFESpace)

Retrieve and return the degrees of freedom (d.o.f.s) partition for `space`.

# Arguments
- `space::AbstractFESpace`: Single-component finite element space.

# Returns
- `::Vector{Vector{Vector{Int}}}`: Nested d.o.f. partition. The first level of nesting
    corresponds to the patch. The second level corresponds to the division (see ... for its
    definition). The third level corresponds to the individual d.o.f.s.
"""
function get_dof_partition(space::AbstractFESpace)
    return space.dof_partition
end

"""
    get_max_local_dim(space::AbstractFESpace)

Compute an upper bound of the element-local dimension of `space`. Note that this is not
necessarily a tight upper bound.

The fallback computation for multi-component spaces is a very conservative estimate to
avoid having to loop over all elements, computing the union of all active basis functions,
and returning the length of the largest union.

For single-component spaces, there is no general fallback.

# Arguments
- `space::AbstractFESpace`: A finite element space.

# Returns
- `::Int`: The element-local upper bound.

# Exceptions
- Error "'get_max_local_dim' not implemented": This error is thrown if no 'get_max_local_dim'
    method is defined for a single-component space (which can be a component of a multi-
    component space).
"""
function get_max_local_dim(space::AbstractFESpace)
    max_local_dim = 0
    for space in space.component_spaces
        max_local_dim += max(max_local_dim, get_max_local_dim(space))
    end
    return max_local_dim
end

function get_max_local_dim(
    space::AbstractFESpace{manifold_dim, 1, num_patches}
) where {manifold_dim, num_patches}
    return error("'get_max_local_dim' not implemented for $(typeof(space))")
end

"""
    get_num_elements(space::AbstractFESpace)

Returns the total number of elements on which the `space` is build.

# Arguments
- `space::AbstractFESpace`: A finite element space.

# Returns
- `::Int`: The number of elements.
"""
function get_num_elements(space::AbstractFESpace)
    return get_num_elements(get_extraction_operator(space))
end

# Getting the element lengths or measure should be handled by the underlying geometry. With
# the upcoming rewrite and reordering of geometry and function spaces, this should make more
# sense.
# """
#     get_element_lengths(space::AbstractFESpace, element_id::Int)

# Returns the size of the element specified by `element_id`.

# # Arguments
# - `space::AbstractFESpace`: The finite element space.
# - `element_id::Int`: The global id of the element.

# # Returns
# - `::Float64`: The size of the element.
# """
# function get_element_lengths(space::AbstractFESpace, element_id::Int)
#     patch_id, local_element_id = get_patch_and_local_element_id(space, element_id)

#     return get_element_lengths(get_patch_space(space, patch_id), local_element_id)
# end

"""
    get_local_basis(
        space::AbstractFESpace{manifold_dim, num_components, num_patches},
        element_id::Int,
        xi::NTuple{manifold_dim,Vector{Float64}},
        nderivatives::Int,
        component_id::Int=1,
    ) where {manifold_dim, num_components, num_patches}

Evaluates the local basis functions used to construct the finite element space `space`. The
default for multi-component spaces are the component spaces. There is no default for single-
component spaces. Single-component spaces are expected to have a specialised implementation.

# Arguments
- `space::AbstractFESpace{manifold_dim, num_components, num_patches}`: A finite element
    space.
- `element_id::Int`: The indentifier of the element.
- `xi::Points.AbstractPoints{manifold_dim}`: The coordinates at which to evaluate the
    basis functions. These coordinates are in the **canonical** domain, and thus always lie
    in the interval `[0, 1]`. The coordinates have to be given per dimension. Multi-
    dimensional spaces are evaluated on the tensor product of the given coordinates.
- `nderivatives::Int`: The number of derivatives to compute.
- `component_id::Int=1`: The component ID. This is only relevant for multi-component
    spaces, and thus defaults to 1. While it is not needed for single-component spaces, it
    is still required for the function signature.

# Returns
- `::Vector{Vector{Vector{Matrix{Float64}}}}`: A nested vector structure containing the
    local basis functions and their derivatives. The first level of nesting corresponds to
    the order of the derivatives. The second level corresponds to a specific derivative. The
    third level corresponds to the component. The matrix contains the actual evaluations.
    See [`get_derivative_idx`](@ref Mantis.GeneralHelpers.get_derivative_idx) for more details
    on the order in which the derivatives are stored.
"""
function get_local_basis(
    space::AbstractFESpace{manifold_dim, num_components, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
    component_id::Int=1,
) where {manifold_dim, num_components, num_patches}
    vals, _ = evaluate(
        get_component_spaces(space)[component_id], element_id, xi, nderivatives
    )

    return vals
end

"""
    evaluate(
        space::AbstractFESpace{manifold_dim, num_components, num_patches},
        element_id::Int,
        xi::NTuple{manifold_dim,Vector{Float64}},
        nderivatives::Int=0
    ) where {manifold_dim, num_components, num_patches}

Evaluate the basis functions of the finite element space `space` at the point `xi` on
element `element_id` up to order `nderivatives`.

All of our function spaces are built in the parameteric domain. For example, in 1D, let the
mapping from canonical coordinates to the `i`-th parametric element be
`Φᵢ: [0, 1] -> [aᵢ, bᵢ]` then, for a function `f` defined on the parametric patch, the
evaluate method actually returns the values/derivatives of `f∘Φᵢ`. Consider the function
`f₀ := f∘Φᵢ`. Then, `FunctionSpaces.evaluate(f)` returns
`evaluations::Vector{Vector{Vector{Matrix{Float64}}}` where `evaluations[i][j][k][a,b]` is
the evaluation of:
- the `j`-th mixed derivative ...
- of order `i-1` ...
- for the `b`-th basis function ...
- of the `k`-th component ...
- at the `a`-th evaluation point ...
- for `f₀`.
See [`get_derivative_idx`](@ref GeneralHelpers.get_derivative_idx) for more details on
the order in which all the mixed derivatives of order `i-1` are stored.

# Arguments
- `space::AbstractFESpace{manifold_dim, num_components, num_patches}`: Finite element space.
- `element_id::Int`: Index of the element.
- `xi::Points.AbstractPoints{manifold_dim}`: Point on the element in canonical coordinates.
- `nderivatives::Int=0`: Order of the derivatives. Default is 0 (i.e., function evaluation).

# Returns
- `evaluation::Vector{Vector{Vector{Matrix{Float64}}}}`: Values of the basis functions.
- `basis_indices::TI`: Global indices of the basis functions. The type `TI` is an vector-
    like object with integer type elements. See the documentation of `space` or
    [`Indices`](@ref) for more details.
"""
function evaluate(
    space::AbstractFESpace{manifold_dim, num_components, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int=0,
) where {manifold_dim, num_components, num_patches}
    basis_indices = get_basis_indices(space, element_id)
    # Pre-allocation.
    num_points = Points.get_num_points(xi)
    evaluations = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
    for j in 0:nderivatives
        # number of derivatives of order j
        num_j_ders = binomial(manifold_dim + j - 1, manifold_dim - 1)
        evaluations[j + 1] = Vector{Vector{Matrix{Float64}}}(undef, num_j_ders)
        for der_idx in 1:num_j_ders
            evaluations[j + 1][der_idx] = [
                zeros(num_points, length(basis_indices)) for _ in 1:num_components
            ]
        end
    end

    # Evaluate the basis functions.
    for component_idx in 1:num_components
        extraction_coefficients, J = get_extraction(space, element_id, component_idx)
        component_basis = get_local_basis(
            space, element_id, xi, nderivatives, component_idx
        )

        for der_order in eachindex(evaluations)
            for der_idx in eachindex(evaluations[der_order])
                LinearAlgebra.mul!(
                    view(evaluations[der_order][der_idx][component_idx], :, J),
                    component_basis[der_order][der_idx][1],
                    extraction_coefficients,
                )
            end
        end
    end

    return evaluations, basis_indices
end

"""
    evaluate(
        space::AbstractFESpace{manifold_dim, num_components, num_patches},
        element_id::Int,
        xi::NTuple{manifold_dim,Vector{Float64}},
        nderivatives::Int,
        coefficients::Vector{Float64}
    ) where {manifold_dim, num_components, num_patches}

Evaluate the basis functions with coefficients of the finite element space `space` at the
point `xi` on element `element_id` up to order `nderivatives`. See [`evaluate`](@ref) for
more details.


# Arguments
- `space::AbstractFESpace{manifold_dim, num_components, num_patches}`: Finite element space.
- `element_id::Int`: Index of the element.
- `xi::Points.AbstractPoints{manifold_dim}`: Point on the element in canonical coordinates.
- `nderivatives::Int=0`: Order of the derivatives. Default is 0 (i.e., function evaluation).
- `coefficients::Vector{Float64}`: Coefficients.

# Returns
- `evaluation::Vector{Vector{Vector{Vector{Float64}}}}`: Values at the points.
"""
function evaluate(
    space::AbstractFESpace{manifold_dim, num_components, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
    coefficients::Vector{Float64},
) where {manifold_dim, num_components, num_patches}
    basis_indices = get_basis_indices(space, element_id)

    basis_evaluations, _ = evaluate(space, element_id, xi, nderivatives)

    evaluations = Vector{Vector{Vector{Vector{Float64}}}}(undef, nderivatives + 1)
    for j in 0:nderivatives
        # number of derivatives of order j
        num_j_ders = binomial(manifold_dim + j - 1, manifold_dim - 1)
        evaluations[j + 1] = Vector{Vector{Vector{Float64}}}(undef, num_j_ders)
        for der_idx in 1:num_j_ders
            for component_idx in 1:num_components
                evaluations[j + 1][der_idx] = [
                    basis_evaluations[j + 1][der_idx][component_idx] *
                    coefficients[basis_indices],
                ]
            end
        end
    end

    return evaluations
end

"""
    assemble_global_extraction_matrix(space::AbstractFESpace)

Assembles the global extraction matrix for an FESpace by combining the extraction
coefficients from each element on each patch. The global extraction matrix maps the global
degrees of freedom to the local degrees of freedom.

# Arguments
- `space::AbstractFESpace`: The finite element space.

# Returns
- `SparseMatrixCSC{Float64}`: The global extraction matrix that maps global dofs to local dofs
"""
function assemble_global_extraction_matrix(space::AbstractFESpace)
    throw(
        ArgumentError(
            "'assemble_global_extraction_matrix' not implemented for $(typeof(space))"
        ),
    )
end

"""
    _evaluate_all_at_point(
        fem_space::AbstractFESpace{1, num_components},
        element_id::Int,
        xi::Float64,
        nderivatives::Int,
    ) where {num_components}

Evaluates all derivatives up to order `nderivatives` for all basis functions of `fem_space`
at a given point `xi` in the element `element_id`.

# Arguments
- `fem_space::AbstractFESpace{1, num_components}`: A univariate FEM space.
- `element_id::Int`: The id of the element.
- `xi::Float64`: The point where all global basis functiuons are evaluated.
- `nderivatives::Int`: The order upto which derivatives need to be computed.

# Returns
- `::SparseMatrixCSC{Float64}`: Global basis functions, size = n_dofs x nderivatives+1
"""
function _evaluate_all_at_point(
    fem_space::AbstractFESpace{1, num_components, num_patches},
    element_id::Int,
    xi::Float64,
    nderivatives::Int,
) where {num_components, num_patches}
    basis_eval, basis_indices = evaluate(
        fem_space, element_id, Points.CartesianPoints(([xi],)), nderivatives
    )
    nloc = length(basis_indices)
    ndofs = get_num_basis(fem_space)
    I = zeros(Int, nloc * (nderivatives + 1))
    J = zeros(Int, nloc * (nderivatives + 1))
    V = zeros(Float64, nloc * (nderivatives + 1))
    count = 0
    for r in 0:nderivatives
        for i in 1:nloc
            I[count + 1] = basis_indices[i]
            J[count + 1] = r + 1
            V[count + 1] = basis_eval[r + 1][1][1][1, i]
            count += 1
        end
    end

    return SparseArrays.sparse(I, J, V, ndofs, nderivatives + 1)
end

"""
    _compute_parametric_geometry_coeffs(fem_space::F) where {
        manifold_dim,
        num_components,
        F<:FunctionSpaces.AbstractFESpace{manifold_dim, num_components}
    }

Compute the parametric geometry coefficients for a given finite element space.

This function calculates the coefficients necessary for representing the parametric geometry
of a finite element space.

# Arguments
- `fem_space::F`: The finite element space for which to compute the parametric geometry
    coefficients.

# Returns
- `coeffs::Matrix{Float64}`: The coefficients necessary for representing the parametric
    geometry. The size of the matrix is (num_basis x manifold_dim).
"""
function _compute_parametric_geometry_coeffs(
    fem_space::F
) where {manifold_dim, num_components, F <: AbstractFESpace{manifold_dim, num_components}}
    function _compute_element_xi(xi_canonical_prod, element_vertices)
        element_xi = Matrix{Float64}(undef, size(xi_canonical_prod))

        # affine transformation x ↦ a+x(b-a)
        for dim in axes(xi_canonical_prod, 2)
            element_xi[:, dim] = @. element_vertices[dim][1] +
                xi_canonical_prod[:, dim] *
                                    (element_vertices[dim][2] - element_vertices[dim][1])
        end

        return element_xi
    end
    degrees = get_polynomial_degree(fem_space)

    num_xi_per_dim = degrees .+ 1
    num_xi = prod(num_xi_per_dim)

    xi_canonical = tuple(
        collect.(range(0, 1, num_xi_per_dim[i]) for i in 1:manifold_dim)...
    )
    xi_canonical_prod = Matrix{Float64}(undef, num_xi, manifold_dim)
    for (id, point) in enumerate(Iterators.product(xi_canonical...))
        xi_canonical_prod[id, :] .= point
    end

    num_elements = get_num_elements(fem_space)

    points = Matrix{Float64}(undef, num_elements * num_xi, manifold_dim)
    A = zeros(size(points, 1), get_num_basis(fem_space))

    for element_id in 1:num_elements
        element_vertices = get_element_vertices(fem_space, element_id)
        element_xi = _compute_element_xi(xi_canonical_prod, element_vertices)

        curr_points_ids = ((element_id - 1) * num_xi + 1):(element_id * num_xi)
        points[curr_points_ids, :] = element_xi
        eval = evaluate(fem_space, element_id, xi_canonical)
        A[curr_points_ids, eval[2]] .= eval[1][1][1]
    end

    coeffs = A \ points

    return coeffs
end

include("ExtractionOperator.jl")

include("OtherSpaces/RationalFESpaces.jl")
include("OtherSpaces/DirectSumSpace.jl")

include("UnivariateSplines/UnivariateSplines.jl")
include("TensorProductSpaces/TensorProductSpaces.jl")

include("UnstructuredSpaces/GTBSplines.jl")
include("UnstructuredSpaces/PolarSplines.jl")

include("TwoScaleRelations/AbstractTwoScaleRelations.jl")

include("Hierarchical/Hierarchical.jl")
