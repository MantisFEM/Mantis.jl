############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    FormSpace{manifold_dim, form_rank, F, L} <: AbstractFormSpace{manifold_dim, form_rank}

Differential forms with a basis.

A `FormSpace` relies on a [`FunctionSpaces.AbstractFESpace`](@ref) to represent a
differential form with the function space as basis. While the function space provides a
basis, the `form_rank` of the `FormSpace` will dictate the behaviour of the form (i.e. is
it a ``0``-form, ``1``-form, etc.) and thus its properties.

# Constructors
- `FormSpace(form_rank::Int, fem_space::F, label::AbstractString)`: General constructor.

# Example
```jldoctest
julia> using Mantis

julia> B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (4, 4), (3, 3), (2,2));

julia> Λ⁰ₕ = Forms.FormSpace(0, B, "0-form");  # 0-form with B as basis.

julia> Λ²ₕ = Forms.FormSpace(2, B, "2-form");  # 2-form with B as basis.
```

# Fields
- `fem_space::F`: The finite element space [`FunctionSpaces.AbstractFESpace`](@ref) used as
    basis for this form. From this space, the `manifold_dim` and geometry are inherited.
    Additionally, the `num_components` of the function space must be consistent with the
    provided `form_rank` and the `manifold_dim`, i.e., a real-valued ``0``-form has 1
    component (in any dimension), a ``1``-form in 3D has 3 components, etc.
- `label::AbstractString`: Label for the form space. This will be used in export and
    plotting functions to easily identify the form.

# Type parameters
- `manifold_dim`, `form_rank`, `expression_rank`: See [`AbstractForm`](@ref) for the details.
- `F`: Type of the finite element space (a [`FunctionSpaces.AbstractFESpace`](@ref)).
- `L`: Type of the label (an `AbstractString`).
"""
struct FormSpace{manifold_dim, form_rank, F, L} <:
       AbstractFormSpace{manifold_dim, form_rank}
    fem_space::F
    label::L

    function FormSpace(
        form_rank::Int, fem_space::F, label::AbstractString
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        F <: FunctionSpaces.AbstractFESpace{manifold_dim, num_components, num_patches},
    }
        if (form_rank ∈ Set([0, manifold_dim])) && (num_components > 1)
            throw(
                ArgumentError(
                    "Mantis.Forms.FormSpace: form_rank = $form_rank with " *
                    "manifold_dim = $manifold_dim requires an FE space with only one " *
                    "component (got num_components = $num_components).",
                ),
            )
        elseif (form_rank ∉ Set([0, manifold_dim])) && (num_components != manifold_dim)
            throw(
                ArgumentError(
                    "Mantis.Forms.FormSpace: form_rank = $form_rank with " *
                    "manifold_dim = $manifold_dim requires an FE space with " *
                    "num_components = $manifold_dim (got $num_components).",
                ),
            )
        end

        return new{manifold_dim, form_rank, F, typeof(label)}(fem_space, label)
    end
end

############################################################################################
#                                   Getters and setters                                    #
############################################################################################

get_form(form_space::FormSpace) = form_space

get_form_space_tree(form_space::FormSpace) = (get_form(form_space),)

get_estimated_nnz_per_elem(form_space::FormSpace) = get_max_local_dim(form_space)

get_geometry(form_space::FormSpace) = FunctionSpaces.get_geometry(get_fe_space(form_space))

function get_num_basis(form_space::FormSpace)
    return FunctionSpaces.get_num_basis(get_fe_space(form_space))
end

function get_num_basis(form_space::FormSpace, element_id::Int)
    return FunctionSpaces.get_num_basis(get_fe_space(form_space), element_id)
end

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(
    form_space::FormSpace{manifold_dim, form_rank},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank}
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
        form_space::FormSpace{manifold_dim, form_rank},
        element_idx::Int,
        xi::Points.AbstractPoints{manifold_dim},
        nderivatives::Int,
    ) where {manifold_dim, form_rank}

Evaluate the form basis functions and their arbitrary derivatives in canonical coordinates.

# Arguments
- `form_space::FormSpace{manifold_dim, form_rank}`: The form space.
- `element_idx::Int`: Index of the element where the evaluation is performed.
- `xi::Points.AbstractPoints{manifold_dim}`: Canonical points for evaluation.

# Returns
- `local_form_basis::Vector{Vector{Vector{Matrix{Float64}}}}`: The basis functions evaluated
    at the canonical coordinates of the element.
- `::Vector{Vector{Int}}`: The basis functions evaluated at the canonical coordinates of the
    element.
"""
function _evaluate_form_in_canonical_coordinates(
    form_space::FormSpace{manifold_dim, form_rank},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
) where {manifold_dim, form_rank}
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
