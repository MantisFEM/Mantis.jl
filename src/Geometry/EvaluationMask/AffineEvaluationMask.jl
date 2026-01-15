############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    AffineEvaluationMask{manifold_dim, num_elements, num_elements_base, M, T, S}

Given an object with d-dimensional elements 1:n, we construct:
* A new object with d-dimensional elements 1:m
* A function E: {1:m} -> {1:n} such that if E(i) = j, then the i-th element is nested
inside the j-th element
* A data structure F: {1:m} -> (x, c) \\in [0,1]^{2d} such that, after rescaling the j-th
 element to a unit cell, a point y inside the i-th element can be located inside the j-th
 element as: x + c * y.
The tuple (E, F) defines an evaluation mask.

# Fields
- `element_id_map::M`: The mapping from the evaluation elements to the base elements.
- `translations::Vector{T}`: The translations from base to evaluation elements.
- `scalings::Vector{S}`: The scalings of evalution elements relative to the base elements.
"""
struct AffineEvaluationMask{manifold_dim, M, T, S} <: AbstractEvaluationMask{manifold_dim}
    element_id_map::M
    num_elements::Int
    num_elements_base::Int
    translations::T
    scalings::S

    function AffineEvaluationMask(
        manifold_dim::Int,
        num_elements::Int,
        num_elements_base::Int,
        element_id_map::M,
        translations::T,
        scalings::S,
    ) where {M <: Function, T <: Function, S <: Function}
        return new{manifold_dim, M, T, S}(
            element_id_map, num_elements, num_elements_base, translations, scalings
        )
    end
end

function AffineEvaluationMask(
    num_elements::Int,
    num_elements_base::Int,
    element_id_map::M,
    translations::T,
    scalings::S,
) where {
    manifold_dim,
    TT <: NTuple{manifold_dim, Real},
    ST <: NTuple{manifold_dim, Real},
    M <: Union{Function, AbstractVector{Int}},
    T <: Union{Function, Vector{TT}},
    S <: Union{Function, Vector{ST}},
}
    if !(M <: Function)
        if length(element_id_map) != num_elements
            throw(
                ArgumentError(
                    "The number of element mappings must match the number of elements."
                ),
            )
        end

        element_id_func(element_id::Int) = element_id_map[element_id]
    else
        element_id_func = element_id_map
    end

    if !(T <: Function)
        if length(translations) != num_elements
            throw(
                ArgumentError(
                    "The number of translations must match the number of elements."
                ),
            )
        end

        translations_func(element_id::Int) = translations[element_id]
    end

    if !(S <: Function)
        if length(scalings) != num_elements
            throw(
                ArgumentError("The number of scalings must match the number of elements.")
            )
        end

        scalings_func(element_id::Int) = scalings[element_id]
    end

    return AffineEvaluationMask(
        manifold_dim,
        num_elements,
        num_elements_base,
        element_id_func,
        translations_func,
        scalings_func,
    )
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_element_id_map(eval_mask::AffineEvaluationMask) = eval_mask.element_id_map
get_translations(eval_mask::AffineEvaluationMask) = eval_mask.translations
get_scalings(eval_mask::AffineEvaluationMask) = eval_mask.scalings

"""
    get_base_element(eval_mask::AffineEvaluationMask, element_id::Int)

Get the base element index.

# Arguments
- `eval_mask::AffineEvaluationMask`: The evaluation mask.
- `element_id::Int`: The element index.

# Returns
- `element_id_base::Int`: The base element index.
"""
function get_base_element(eval_mask::AffineEvaluationMask, element_id::Int)
    return get_element_id_map(eval_mask)(element_id)
end

"""
    get_translation(eval_mask::AffineEvaluationMask, element_id::Int)

Get the translation of the element.

# Arguments
- `eval_mask::AffineEvaluationMask`: The evaluation mask.
- `element_id::Int`: The element index.

# Returns
- `translation<:NTuple{manifold_dim, Real}`: The translation.
"""
function get_translation(eval_mask::AffineEvaluationMask, element_id::Int)
    return get_translations(eval_mask)(element_id)
end

"""
    get_measure_scale(eval_mask::AffineEvaluationMask, element_id::Int)

Get the product of the scaling of the element.

# Arguments
- `eval_mask::AffineEvaluationMask`: The evaluation mask.
- `element_id::Int`: The element index.

# Returns
- `scaling::Float64`: The scaling.
"""
function get_measure_scale(eval_mask::AffineEvaluationMask, element_id::Int)
    return prod(get_scaling(eval_mask, element_id))
end

"""
    get_scaling(eval_mask::AffineEvaluationMask, element_id::Int)

Get the scaling of the element.

# Arguments
- `eval_mask::AffineEvaluationMask`: The evaluation mask.
- `element_id::Int`: The element index.

# Returns
- `scaling<:NTuple{manifold_dim, Real}`: The scaling.
"""
function get_scaling(eval_mask::AffineEvaluationMask, element_id::Int)
    return get_scalings(eval_mask)(element_id)
end

############################################################################################
#                                        Evaluation                                        #
############################################################################################

"""
    transform_evaluation_points(
        eval_mask::AffineEvaluationMask{manifold_dim, num_elements, num_elements_base},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim}
    )

Evaluate the mapping from the evaluation mask to the base.

# Arguments
- `eval_mask::AffineEvaluationMask`: The evaluation mask.
- `element_id::Int`: The evaluation element index.
- `xi::Points.AbstractPointsP{manifold_dim}`: The canonical points for the evaluation mask.

# Returns
- `element_id_base::Int`: The element index in the range geometry.
- `xi_base:Points.AbstractPoints{manifold_dim}`: The canonical points for the base.
- `scaling::NTuple{manifold_dim, Float64}`: The scaling of the element.
"""
function transform_evaluation_points(
    eval_mask::AffineEvaluationMask{manifold_dim, num_elements, num_elements_base},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, num_elements, num_elements_base}
    element_id_base = get_base_element(eval_mask, element_id)
    scaling = get_scaling(eval_mask, element_id)
    xi_base = Points.scale_and_shift_points(
        xi, scaling, get_translation(eval_mask, element_id)
    )

    return element_id_base, xi_base, scaling
end

function transform_element_vertices(
    eval_mask::AffineEvaluationMask{manifold_dim},
    element_vertices_base::NTuple{manifold_dim, NTuple{2, T}},
    element_id::Int,
) where {manifold_dim, T <: Real}
    translation = get_translation(eval_mask, element_id)
    scaling = get_scaling(eval_mask, element_id)
    element_vertices = ntuple(
        dim -> element_vertices_base[dim] .* scaling[dim] .+ translation[dim], manifold_dim
    )

    return element_vertices
end

"""
    get_element_ids(eval_mask::AffineEvaluationMask, element_id_base::Int)

Get the element indices corresponding to a base element.

# Arguments
- `eval_mask::AffineEvaluationMask`: The evaluation mask.
- `element_id_base::Int`: The base element index.

# Returns
- `element_ids::Vector{Int}`: The element indices.
"""
function get_element_ids(eval_mask::AffineEvaluationMask, element_id_base::Int)
    return findall(id -> id == element_id_base, get_element_id_map(eval_mask))
end

"""
    compose_evaluation_masks(
		eval_mask_1::AffineEvaluationMask{manifold_dim, num_elements, num_elements_base_1},
		eval_mask_2::AffineEvaluationMask{
			manifold_dim, num_elements_base_1, num_elements_base_2
		},
    )

Compose two evaluation maskes.

# Arguments
- `eval_mask_1::AffineEvaluationMask`: The first evaluation mask.
- `eval_mask_2::AffineEvaluationMask`: The second evaluation mask

# Returns
- `eval_mask::AffineEvaluationMask`: The composed evaluation mask (E_2\\circ E_1), such that
the evaluation elements are nested inside the base elements of the second evaluation mask.
"""
function compose_evaluation_masks(
    eval_mask_1::AffineEvaluationMask{manifold_dim},
    eval_mask_2::AffineEvaluationMask{manifold_dim},
) where {manifold_dim}
    num_elements_1 = get_num_elements(eval_mask_1)
    num_elements_base_1 = get_num_elements_base(eval_mask_1)
    num_elements_2 = get_num_elements(eval_mask_2)
    num_elements_base_2 = get_num_elements_base(eval_mask_2)
    if num_elements_base_1 != num_elements_2
        msg =
            "The number of base elements in the first mask should match " *
            "the total number of elements in the second mask. " *
            "The given numbers were $(num_elements_base_1) and $(num_elements_2)."
        throw(ArgumentError(msg))
    end

    element_id_map = [
        get_base_element(eval_mask_2, get_base_element(eval_mask_1, i)) for
        i in 1:num_elements_1
    ]
    scalings = [
        get_scaling(eval_mask_2, get_base_element(eval_mask_1, i)) .*
        get_scaling(eval_mask_1, i) for i in 1:num_elements_1
    ]
    translations = [
        get_translation(eval_mask_2, get_base_element(eval_mask_1, i)) .+
        get_scaling(eval_mask_2, get_base_element(eval_mask_1, i)) .*
        get_translation(eval_mask_1, i) for i in 1:num_elements_1
    ]

    return AffineEvaluationMask(
        num_elements_1, num_elements_base_2, element_id_map, translations, scalings
    )
end

"""
    trivial_evaluation_mask(manifold_dim::Int, num_elements::Int)

Create a trivial evaluation mask.

# Arguments
- `manifold_dim::Int`: The dimension of the manifold.
- `num_elements::Int`: The number of elements.

# Returns
- `eval_mask::AffineEvaluationMask`: The trivial evaluation mask.
"""
function trivial_evaluation_mask(manifold_dim::Int, num_elements::Int)
    element_id_map(element_id) = element_id
    translations(_) = ntuple(_ -> 0, manifold_dim)
    scalings(_) = ntuple(_ -> 1, manifold_dim)

    return AffineEvaluationMask(
        manifold_dim, num_elements, num_elements, element_id_map, translations, scalings
    )
end

"""
    trim_evaluation_mask(
        eval_mask::AffineEvaluationMask{manifold_dim, num_elements, num_elements_base},
        elements_to_exclude::AbstractVector{Int}
    )

Trim the evaluation mask by excluding elements.

# Arguments
- `eval_mask::AffineEvaluationMask`: The evaluation mask.

# Returns
- `eval_mask::AffineEvaluationMask`: The trimmed evaluation mask.
"""
function trim_evaluation_mask(
    eval_mask::AffineEvaluationMask{manifold_dim, num_elements, num_elements_base},
    elements_to_exclude::AbstractVector{Int},
) where {manifold_dim, num_elements, num_elements_base}
    num_new_elements = num_elements - length(elements_to_exclude)
    element_id_map = zeros(Int, num_new_elements)
    count = 1
    translations = Vector{NTuple{manifold_dim, Float64}}(undef, num_new_elements)
    scalings = Vector{NTuple{manifold_dim, Float64}}(undef, num_new_elements)
    for i in 1:num_elements
        if i in elements_to_exclude
            continue
        end

        element_id_map[count] = get_base_element(eval_mask, i)
        translations[count] = get_translation(eval_mask, i)
        scalings[count] = get_scaling(eval_mask, i)
        count += 1
    end

    return AffineEvaluationMask(
        num_new_elements, num_elements_base, element_id_map, translations, scalings
    )
end
