"""
    get_dorfler_marking(element_errors::Vector{Float64}, dorfler_parameter::Float64)

Computes the indices of elements with at least `dorfler_parameter*100`% of the highest error in `element_errors`.

# Arguments
- `element_errors::Vector{Float64}`: element-wise errors.
- `dorfler_parameter::Float64`: dorfler parameter determing how many elements are selected.

# Returns
- `::Vector{Int}`: indices of elements with at least `dorfler_parameter*100`% of the highest error.
"""
function get_dorfler_marking(element_errors::Vector{Float64}, dorfler_parameter::Float64)
    0.0 <= dorfler_parameter < 1.0 || throw(
        ArgumentError(
            "Dorfler parameter should be between 0 and 1. The given value was $dorfler_parameter.",
        ),
    )

    max_error = maximum(element_errors)

    return findall(error -> error > (1.0 - dorfler_parameter) * max_error, element_errors)
end

function add_padding!(
    marked_elements_per_level::Vector{Vector{Int}},
    hier_space::HierarchicalFiniteElementSpace,
)
    num_levels = get_num_levels(hier_space)

    for level in 1:num_levels
        if marked_elements_per_level[level] == Int[]
            continue
        end

        basis_in_marked_elements = reduce(
            union,
            get_basis_indices.(
                Ref(hier_space.spaces[level]), marked_elements_per_level[level]
            ),
        )
        marked_elements_per_level[level] = reduce(
            union, get_support.(Ref(hier_space.spaces[level]), basis_in_marked_elements)
        )
    end

    return marked_elements_per_level
end

function add_padding!(
    marked_elements_per_level::Vector{Vector{Int}}, spaces::Vector{S}
) where {manifold_dim, S <: AbstractFESpace{manifold_dim, 1}}
    num_levels = length(spaces)

    for level in 1:num_levels
        if marked_elements_per_level[level] == Int[]
            continue
        end

        basis_in_marked_elements = reduce(
            union, get_basis_indices.(Ref(spaces[level]), marked_elements_per_level[level])
        )
        marked_elements_per_level[level] = union(
            get_support.(Ref(spaces[level]), basis_in_marked_elements)...
        )
    end

    return marked_elements_per_level
end

function get_marked_elements_children(
    hier_space::HierarchicalFiniteElementSpace{
        manifold_dim, num_components, num_patches, S, T
    },
    marked_elements_per_level::Vector{Vector{Int}},
    new_operator::T,
) where {manifold_dim, num_components, num_patches, S, T <: AbstractTwoScaleOperator}
    num_levels = get_num_levels(hier_space)

    marked_children = Vector{Vector{Int}}(undef, num_levels)
    marked_children[1] = Int[]

    for level in 1:num_levels
        if level < num_levels
            if marked_elements_per_level[level] == Int[]
                marked_children[level + 1] = Int[]
            else
                marked_children[level + 1] = reduce(
                    vcat,
                    get_element_children.(
                        Ref(get_twoscale_operator(hier_space, level)),
                        marked_elements_per_level[level],
                    ),
                )
            end
        elseif marked_elements_per_level[level] != Int[]
            push!(
                marked_children,
                reduce(
                    vcat,
                    get_element_children.(
                        Ref(new_operator), marked_elements_per_level[level]
                    ),
                ),
            )
        end
    end

    return marked_children
end

function get_padding_per_level(
    hier_space::HierarchicalFiniteElementSpace, marked_elements::Vector{Int}
)
    element_ids_per_level = convert_element_vector_to_elements_per_level(
        hier_space, marked_elements
    )
    add_padding!(element_ids_per_level, hier_space)

    return element_ids_per_level
end

function get_refinement_domains(
    hier_space::HierarchicalFiniteElementSpace{
        manifold_dim, num_components, num_patches, S, T
    },
    marked_elements_per_level::Vector{Vector{Int}},
    new_operator::T,
) where {manifold_dim, num_components, num_patches, S, T <: AbstractTwoScaleOperator}
    return get_marked_elements_children(hier_space, marked_elements_per_level, new_operator)
end
