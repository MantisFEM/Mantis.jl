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
- `manifold_dim`, `form_rank`, `expression_rank`: See [`AbstractForm`](@ref).
- `PB`: Type of the pullback used, see [`AbstractPullback`](@ref).
- `S`: Type of the domain used, see [`AbstractPullbackLocation`](@ref).
- `F`: Type of the finite element space (a [`FunctionSpaces.AbstractFESpace`](@ref)).
- `L`: Type of the label (an `AbstractString`).
"""
struct FormSpace{manifold_dim, form_rank, PB, S, F, L} <:
       AbstractFormSpace{manifold_dim, form_rank}
    fem_space::F
    label::L

    function FormSpace(
        form_rank::Int,
        fem_space::F,
        label::AbstractString,
        ::Type{PB}=FormPullback,
        ::Type{S}=Parametric,
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        F <: FunctionSpaces.AbstractFESpace{manifold_dim, num_components, num_patches},
        PB,
        S,
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

        return new{manifold_dim, form_rank, PB, S, F, typeof(label)}(fem_space, label)
    end
end

############################################################################################
#                                   Getters and setters                                    #
############################################################################################

get_form(form::FormSpace) = form

get_form_space_tree(form::FormSpace) = (get_form(form),)

get_estimated_nnz_per_elem(form::FormSpace) = get_max_local_dim(form)

get_geometry(form::FormSpace) = FunctionSpaces.get_geometry(get_fe_space(form))

function get_num_basis(form::FormSpace)
    return FunctionSpaces.get_num_basis(get_fe_space(form))
end

function get_num_basis(form::FormSpace, element_id::Int)
    return FunctionSpaces.get_num_basis(get_fe_space(form), element_id)
end

function get_pullback_type(
    form::FormSpace{manifold_dim, form_rank, PB}
) where {manifold_dim, form_rank, PB}
    return PB
end

function get_pullback(
    form::FormSpace{manifold_dim, form_rank, PB, S}, ::Type{D}=Canonical
) where {manifold_dim, form_rank, PB, S, D}
    return PB(form, S, D)
end

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(
    form::FormSpace{manifold_dim, form_rank},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank}
    pb = FormPullback(form, Parametric, Canonical)
    local_form_basis, form_basis_indices = evaluate(pb, element_idx, xi)

    return local_form_basis, form_basis_indices
end

function evaluate(
    form::FormPullback{manifold_dim, form_rank, expression_rank, S, D, F},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, expression_rank, S, D, F <: FormSpace}
    evaluations_fe, basis_indices = FunctionSpaces.evaluate(
        get_fe_space(get_form(form)), element_id, xi, 0
    )

    evaluations = evaluations_fe[1][1]

    pullback!(evaluations, form, element_id, xi)

    return evaluations, [basis_indices]
end
