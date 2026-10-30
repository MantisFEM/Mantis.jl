############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    FormSpace{manifold_dim, form_rank, S, F, L} <:
    AbstractFormSpace{manifold_dim, form_rank, S}

Differential forms with a basis.

A `FormSpace` relies on a [`FunctionSpaces.AbstractFESpace`](@ref) to represent a
differential form with the function space as basis. While the function space provides a
basis, the `form_rank` of the `FormSpace` will dictate the behaviour of the form (i.e. is
it a ``0``-form, ``1``-form, etc.) and thus its properties.

Because a `FormSpace` is build on top of an [`FunctionSpaces.AbstractFESpace`](@ref), the
default source location is the [`Parametric`](@ref) domain.

# Constructors
- `FormSpace(form_rank::Int, fem_space::F, label::AbstractString, ::Type{S}=Parametric)`:
    General constructor.

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
- `manifold_dim`, `form_rank`, `expression_rank`, `S`: See [`AbstractForm`](@ref).
- `F`: Type of the finite element space (a [`FunctionSpaces.AbstractFESpace`](@ref)).
- `L`: Type of the label (an `AbstractString`).
"""
struct FormSpace{manifold_dim, form_rank, S, F, L} <:
       AbstractFormSpace{manifold_dim, form_rank, S}
    fem_space::F
    label::L

    function FormSpace(
        form_rank::Int, fem_space::F, label::AbstractString, ::Type{S}=Parametric
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        S,
        F <: FunctionSpaces.AbstractFESpace{manifold_dim, num_components, num_patches},
    }
        if (form_rank == 0 || form_rank == manifold_dim) && (num_components > 1)
            throw(
                ArgumentError(
                    "Mantis.Forms.FormSpace: form_rank = $form_rank with " *
                    "manifold_dim = $manifold_dim requires an FE space with only one " *
                    "component (got num_components = $num_components).",
                ),
            )
        elseif (form_rank != 0 && form_rank != manifold_dim) &&
            (num_components != manifold_dim)
            throw(
                ArgumentError(
                    "Mantis.Forms.FormSpace: form_rank = $form_rank with " *
                    "manifold_dim = $manifold_dim requires an FE space with " *
                    "num_components = $manifold_dim (got $num_components).",
                ),
            )
        end

        return new{manifold_dim, form_rank, S, F, typeof(label)}(fem_space, label)
    end
end

############################################################################################
#                                   Getters and setters                                    #
############################################################################################

get_form(form::FormSpace) = form

get_form_space_tree(form::FormSpace) = (get_form(form),)

get_estimated_nnz_per_elem(form::FormSpace) = get_max_local_dim(form)

get_geometry(form::FormSpace) = FunctionSpaces.get_geometry(get_fe_space(form))

function get_max_local_dim(form::FormSpace)
    return FunctionSpaces.get_max_local_dim(get_fe_space(form))
end

function get_fe_space(form::FormSpace)
    return form.fem_space
end

function get_num_basis(form::FormSpace)
    return FunctionSpaces.get_num_basis(get_fe_space(form))
end

function get_num_basis(form::FormSpace, element_id::Int)
    return FunctionSpaces.get_num_basis(get_fe_space(form), element_id)
end

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(
    form::FormSpace{manifold_dim, form_rank},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank}
    evaluations_fe, basis_indices = FunctionSpaces.evaluate(
        get_fe_space(get_form(form)), element_id, xi, 0
    )

    return evaluations_fe[1][1], [basis_indices]
end
