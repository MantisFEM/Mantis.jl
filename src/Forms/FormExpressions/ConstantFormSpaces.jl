############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    ConstantFormSpace{manifold_dim, form_rank, G, L} <: AbstractFormSpace{manifold_dim, form_rank}

Concrete implementation of a space for constant differential forms. For instance, this can
be used as a Lagrange multiplier enforcing a zero-average constraint on another differential form.

# Fields
- `geometry::G`: The geometry of the domain
- `label::L`: Label for the constant form space

# Type parameters
- `manifold_dim`: Dimension of the manifold
- `form_rank`: Rank of the differential form
- `G`: Type of the geometry
- `L`: Type of the label

# Inner Constructors
- `ConstantFormSpace(form_rank::Int, geometry::G, label::L)`: Constructor for constant
    differential form spaces.
"""
struct ConstantFormSpace{manifold_dim, form_rank, G, L} <:
       AbstractFormSpace{manifold_dim, form_rank}
    geometry::G
    label::L

    """
        ConstantFormSpace(
            form_rank::Int, geometry::G, label::L
        ) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}, L <: AbstractString}

    Constructor for constant differential form spaces.

    # Arguments
    - `form_rank::Int`: Differential form rank.
    - `geometry::G`: The geometry of the domain.
    - `label::L`: The label of the form space.

    # Returns
    - `ConstantFormSpace{manifold_dim, form_rank, G, L}`: The form space.
    """
    function ConstantFormSpace(
        form_rank::Int, geometry::G, label::AbstractString
    ) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}
        if (form_rank ∉ Set([0, manifold_dim]))
            throw(
                ArgumentError(
                    "Mantis.Forms.ConstantFormSpace: form_rank = $form_rank with " *
                    "manifold_dim = $manifold_dim requires form_rank to be 0 or " *
                    "manifold_dim.",
                ),
            )
        end
        return new{manifold_dim, form_rank, G, typeof(label)}(geometry, label)
    end
end

############################################################################################
#                                   Getters and setters                                    #
############################################################################################

"""
    get_num_basis(::ConstantFormSpace)

Get the number of basis functions for the constant form space.

# Returns
- `Int`: Number of basis functions (1).
"""
function get_num_basis(::ConstantFormSpace)
    return 1
end

"""
    get_num_basis(::ConstantFormSpace, ::Int)

Get the number of basis functions for the constant form space on a specific element.

# Arguments
- `::ConstantFormSpace`: The constant form space.
- `::Int`: The parametric element identifier.

# Returns
- `Int`: Number of basis functions (1).
"""
function get_num_basis(::ConstantFormSpace, ::Int)
    return 1
end

"""
    get_max_local_dim(::ConstantFormSpace)

Get the maximum local dimension of the constant form space.

# Returns
- `Int`: Maximum local dimension (1).
"""
function get_max_local_dim(::ConstantFormSpace)
    return 1
end

"""
    get_estimated_nnz_per_elem(::ConstantFormSpace)

Get the number of non-zero entries per element.

# Returns
- `Int`: Number of non-zeros per element (1).
"""
get_estimated_nnz_per_elem(::ConstantFormSpace) = 1

"""
    get_form(form_space::ConstantFormSpace)

Return the form space itself.

# Arguments
- `form_space::ConstantFormSpace`: The constant form space.

# Returns
- `ConstantFormSpace`: The constant form space.
"""
get_form(form_space::ConstantFormSpace) = form_space

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

"""
    evaluate(
        ::ConstantFormSpace{manifold_dim, 0},
        ::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim}

Evaluate the basis function of a 0-form constant space at given canonical points `xi`.

# Arguments
- `::ConstantFormSpace{manifold_dim, 0}`: The constant 0-form space.
- `::Int`: The parametric element identifier.
- `xi::Points.AbstractPoints{manifold_dim}`: The set of canonical points.

# Returns
- `Vector{Matrix{Float64}}`: Vector containing a single `Matrix{Float64}` of ones of size
    `(n_evaluation_points, 1)`.
- `Vector{Vector{Int}}`: The basis function indices evaluated at the canonical coordinates
    of the element, always equal to `[[1]]` in this case.
"""
function evaluate(
    ::ConstantFormSpace{manifold_dim, 0}, ::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    num_evaluation_points = Points.get_num_points(xi)
    return [ones(Float64, num_evaluation_points, 1)], [[1]]
end

"""
    evaluate(
        form_space::ConstantFormSpace{manifold_dim, manifold_dim},
        element_idx::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim}

Evaluate the basis function of a top-level (volume) constant form space at given canonical points
`xi` mapped to the parametric element given by `element_idx`.

# Arguments
- `form_space::ConstantFormSpace{manifold_dim, manifold_dim}`: The constant volume form space.
- `element_idx::Int`: The parametric element identifier.
- `xi::Points.AbstractPoints{manifold_dim}`: The set of canonical points.

# Returns
- `Vector{Matrix{Float64}}`: Vector containing a single `Matrix{Float64}` of size
    `(n_evaluation_points, 1)`, scaled by the determinant of the Jacobian at each point.
- `Vector{Vector{Int}}`: The basis function indices evaluated at the canonical coordinates
    of the element, always equal to `[[1]]` in this case.
"""
function evaluate(
    form_space::ConstantFormSpace{manifold_dim, manifold_dim, G},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}
    num_evaluation_points = Points.get_num_points(xi)
    J = Geometry.jacobian(form_space.geometry, element_idx, xi)  # Jₖⱼ = ∂Φᵏ\\∂ξⱼ
    form_eval = [ones(Float64, num_evaluation_points, 1)]
    form_eval[1][:] .*= LinearAlgebra.det.(J)

    return form_eval, [[1]]
end
