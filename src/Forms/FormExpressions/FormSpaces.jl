############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    FormSpace{manifold_dim, form_rank, G, F} <:
    AbstractFormSpace{manifold_dim, form_rank, G}

Concrete implementation of a function space for differential forms.

# Fields
- `geometry::G`: The geometry of the manifold
- `fem_space::F`: The finite element space(s) used for the form components
- `label::String`: Label for the form space

# Type parameters
- `manifold_dim`: Dimension of the manifold
- `form_rank`: Rank of the differential form
- `G`: Type of the geometry
- `F`: Type of the finite element space

# Inner Constructors
- `FormSpace(form_rank::Int, geometry::G, fem_space::F, label::String)`: General
    constructor for differential form spaces.
"""
struct FormSpace{manifold_dim, form_rank, G, F} <:
       AbstractFormSpace{manifold_dim, form_rank, G}
    geometry::G
    fem_space::F
    label::String

    """
        FormSpace(
            form_rank::Int, geometry::G, fem_space::F, label::String
        ) where {
            manifold_dim,
            num_components,
            num_patches,
            G <: Geometry.AbstractGeometry{manifold_dim},
            F <: FunctionSpaces.AbstractFESpace{manifold_dim, num_components, num_patches},
        }

    General constructor for differential form spaces.

    # Arguments
    - `form_rank::Int`: Differential form rank.
    - `geometry::G`: The geometry where the form is defined.
    - `fem_space::F`: The function space used to represent the form.
    - `label::String`: The label of the form space.

    # Returns
    - `FormSpace{manifold_dim, form_rank, G, F}`: The FormSpace structure.
    """
    function FormSpace(
        form_rank::Int, geometry::G, fem_space::F, label::String
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        G <: Geometry.AbstractGeometry{manifold_dim},
        F <: FunctionSpaces.AbstractFESpace{manifold_dim, num_components, num_patches},
    }
        if (form_rank ∈ Set([0, manifold_dim])) && (num_components > 1)
            throw(
                ArgumentError(
                    "Mantis.Forms.FormSpace: form_rank = $form_rank with " *
                    "manifold_dim = $manifold_dim requires FE space with only one " *
                    "component (got num_compoents = $num_components).",
                ),
            )
        elseif (form_rank ∉ Set([0, manifold_dim])) && (num_components != manifold_dim)
            throw(
                ArgumentError(
                    "Mantis.Forms.FormSpace: form_rank = $form_rank with " *
                    "manifold_dim = $manifold_dim requires a FE space with " *
                    "num_components = $manifold_dim (got num_components = $num_components).",
                ),
            )
        end

        return new{manifold_dim, form_rank, G, F}(geometry, fem_space, label)
    end
end
############################################################################################
#                                   Getters and setters                                    #
############################################################################################
"""
    get_fe_space(form_space::AbstractFormSpace)

Returns the finite element space associated with the given form space.

# Arguments
- `form_space::AbstractFormSpace`: The form space.

# Returns
- `<:FunctionSpaces.AbstractFESpace`: The finite element space.
"""
get_fe_space(form_space::AbstractFormSpace) = form_space.fem_space

"""
    get_form_space_tree(form_space::AbstractFormSpace)

Returns the list of spaces of forms of `expression_rank > 0` in the tree of the expression.
Since `AbstractFormSpace` has a single form of `expression_rank = 1` it returns a `Tuple` 
with the space of of the `AbstractFormSpace`.

# Arguments
- `form_space::AbstractFormSpace`: The AbstractFormSpace structure.

# Returns
- `Tuple(<:AbstractFormExpression)`: The list of forms present in the tree of the expression, in this case the form space.
"""
function get_form_space_tree(form_space::AbstractFormSpace)
    return (get_fe_space(form_space), )  # return a Tuple with a single element because AbstractFormSpace has only one form of expression_rank = 1
end

"""
    get_num_basis(form_space::AbstractFormSpace)

Returns the number of basis functions of the function space associated with the given form
space.

# Arguments
- `form_space::AbstractFormSpace`: The form space.

# Returns
- `Int`: The number of basis functions of the function space.
"""
function get_num_basis(form_space::AbstractFormSpace)
    return FunctionSpaces.get_num_basis(get_fe_space(form_space))
end

"""
    get_num_basis(form_space::AbstractFormSpace, element_id::Int)

Returns the number of basis functions at the given element of the function space associated
the given form space.

# Arguments
- `form_space::AbstractFormSpace`: The form space.

# Returns
- `Int`: The number of basis functions at the given element.
"""
function get_num_basis(form_space::AbstractFormSpace, element_id::Int)
    return FunctionSpaces.get_num_basis(get_fe_space(form_space), element_id)
end

get_estimated_nnz_per_elem(form_space::FormSpace) = get_max_local_dim(form_space)

"""
    get_max_local_dim(form_space::AbstractFormSpace)

Compute an upper bound of the element-local dimension of `form_space`. Note that this is not
necessarily a tight upper bound.

# Arguments
- `form_space::AbstractFormSpace`: The form space.
# Returns
- `::Int`: The element-local upper bound.
"""
function get_max_local_dim(form_space::AbstractFormSpace)
    return FunctionSpaces.get_max_local_dim(get_fe_space(form_space))
end

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

"""
    evaluate(
        form_space::FS, element_idx::Int, xi::Points.AbstractPoints{manifold_dim}
    ) where {
        manifold_dim,
        form_rank,
        G <: Geometry.AbstractGeometry{manifold_dim},
        FS <: AbstractFormSpace{manifold_dim, form_rank, G},
    }

Evaluate the basis functions of a differential form space at given canonical points `xi`
mapped to the parametric element given by `element_idx`.

# Arguments
- `form_space::FS`: The differential form space.
- `element_idx::Int`: The parametric element identifier.
- `xi::NTuple{manifold_dim, Vector{Float64}`: The set of canonical points.

# Returns
- `Vector{Matrix{Float64}}`: Vector of length equal to the number of components of the form,
    where each entry is a `Matrix{Float64}`  of size `(n_evaluation_points,
    n_basis_functions)`
- `form_basis_indices::Vector{Vector{Int}}`: The basis function indices evaluated at the
    canonical coordinates of the element.
"""
function evaluate(
    form_space::FS, element_idx::Int, xi::Points.AbstractPoints{manifold_dim}
) where {
    manifold_dim,
    form_rank,
    G <: Geometry.AbstractGeometry{manifold_dim},
    FS <: AbstractFormSpace{manifold_dim, form_rank, G},
}
    # The form space is made up of components
    # e.g,
    #   0-forms: single component
    #   1-forms:(dξ₁, dξ₂) (2D)
    #   1-forms:(dξ₁, dξ₂, dξ₃) (3D)
    #   2-forms:(dξ₁dξ₂) (2D)
    #   2-forms:(dξ₂dξ₃, dξ₃dξ₁, dξ₁dξ₂) (3D)
    #   3-forms: single component
    # We use the numbering of the function space.

    # Evaluate the form spaces
    local_form_basis, form_basis_indices = _evaluate_form_in_canonical_coordinates(
        form_space, element_idx, xi, 0
    )  # (only evaluate the basis (0-th order derivative))

    return local_form_basis[1][1], form_basis_indices
end

"""
    _evaluate_form_in_canonical_coordinates(
        form_space::FS,
        element_idx::Int,
        xi::Points.AbstractPoints{manifold_dim},
        nderivatives::Int,
    ) where {
        manifold_dim,
        form_rank,
        G <: Geometry.AbstractGeometry{manifold_dim},
        FS <: AbstractFormSpace{manifold_dim, form_rank, G},
    }

Evaluate the form basis functions and their arbitrary derivatives in canonical coordinates.

# Arguments
- `form_space::FS`: The form space.
- `element_idx::Int`: Index of the element where the evaluation is performed.
- `xi::Points.AbstractPoints{manifold_dim}`: Canonical points for evaluation.

# Returns
- `local_form_basis::Vector{Vector{Vector{Matrix{Float64}}}}`: The basis functions evaluated
    at the canonical coordinates of the element.
- `::Vector{Vector{Int}}`: The basis functions evaluated at the canonical coordinates of the
    element.
"""
function _evaluate_form_in_canonical_coordinates(
    form_space::FS,
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
) where {
    manifold_dim,
    form_rank,
    G <: Geometry.AbstractGeometry{manifold_dim},
    FS <: AbstractFormSpace{manifold_dim, form_rank, G},
}
    # Evaluate the form spaces on parametric domain ...
    local_form_basis, form_basis_indices = FunctionSpaces.evaluate(
        get_fe_space(form_space), element_idx, xi, nderivatives
    )  # (only evaluate the basis (0-th order derivative))
    # ... and account for the transformation from a parametric mesh element to the canonical
    # mesh element
    local_form_basis = _pullback_to_canonical_coordinates(
        get_geometry(form_space), local_form_basis, element_idx, form_rank
    )

    # We need to return form_basis_indices as a vector of vectors to allow for multiple
    # index expressions, like the wedge
    return local_form_basis, [form_basis_indices]
end

"""
    _pullback_to_canonical_coordinates(
        geometry::Geometry.AbstractGeometry{manifold_dim},
        form_evaluations::Vector{Vector{Vector{Matrix{Float64}}}},
        element_idx::Int,
        form_rank::Int,
    ) where {manifold_dim}

Pullback the basis functions to the canonical coordinates of the element.

# Arguments
- `geometry::Geometry.AbstractGeometry{manifold_dim}`: The geometry of the form space.
- `form_evaluations::Vector{Vector{Vector{Matrix{Float64}}}}`: The basis functions evaluated
    at the parametric coordinates.
- `element_idx::Int`: Index of the element to evaluate.
- `form_rank::Int`: Rank of the form.

# Returns
- `form_evaluations::Vector{Vector{Vector{Matrix{Float64}}}}`: The form evaluations
    pulled-back to canonical coordinates.
"""
function _pullback_to_canonical_coordinates(
    geometry::Geometry.AbstractGeometry{manifold_dim},
    form_evaluations::Vector{Vector{Vector{Matrix{Float64}}}},
    element_idx::Int,
    form_rank::Int,
) where {manifold_dim}

    # Pullback the evaluations to the canonical coordinates of the element
    if form_rank > 0
        # Get the element dimensions
        element_dimensions = Geometry.get_element_lengths(geometry, element_idx)
        for i in eachindex(form_evaluations)
            for j in eachindex(form_evaluations[i])
                if form_rank == manifold_dim
                    form_evaluations[i][j][1] .*= prod(element_dimensions)
                elseif form_rank == 1
                    for k in 1:manifold_dim
                        form_evaluations[i][j][k] .*= element_dimensions[k]
                    end
                elseif manifold_dim == 3
                    form_evaluations[i][j][1] .*= prod(element_dimensions[2:3])
                    form_evaluations[i][j][2] .*= prod(element_dimensions[1:2:3])
                    form_evaluations[i][j][3] .*= prod(element_dimensions[1:2])
                else
                    throw(
                        ArgumentError(
                            "Mantis.Forms.evaluate: combination of " *
                            "(form rank, manifold dim) = ($form_rank, $manifold_dim) " *
                            "is not supported.",
                        ),
                    )
                end
            end
        end
    end

    return form_evaluations
end
