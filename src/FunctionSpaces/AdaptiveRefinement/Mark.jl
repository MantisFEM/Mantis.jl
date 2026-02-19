"""
    get_dorfler_marking(element_errors::Vector{Float64}, dorfler_parameter::Float64)

Computes the indices of elements with at least `dorfler_parameter*100`% of the highest error
in `element_errors`.

# Arguments
- `element_errors::Vector{Float64}`: element-wise errors.
- `dorfler_parameter::Float64`: dorfler parameter determing how many elements are selected.

# Returns
- `::Vector{Int}`: indices of elements with at least `dorfler_parameter*100`% of the highest
	error.
"""
function get_dorfler_marking(element_errors::Vector{Float64}, dorfler_parameter::Float64)
    if !(0.0 <= dorfler_parameter < 1.0)
        throw(
            ArgumentError(
                "Dorfler parameter should be between 0 and 1. " *
                "The given value was $dorfler_parameter.",
            ),
        )
    end

    max_error = maximum(element_errors)

    return findall(error -> error > (1.0 - dorfler_parameter) * max_error, element_errors)
end

"""
	add_padding!(
	    marked_elements_per_level::Vector{Vector{Int}}, space::HierarchicalFiniteElementSpace
	)

For each `level` of the hierarchical `space`, adds a padding to
`marked_elements_per_level[level]`. The padding consists of the support of all basis
functions whose supports intersect the original `marked_elements_per_level`.

# Arguments
- `marked_elements_per_level::Vector{Vector{Int}}`: The level-wise indexing of marked
	elements.
- `space::HierarchicalFiniteElementSpace`: The hierarchical finite element space.

# Returns
- `marked_elements_per_level::Vector{Vector{Int}}`: The padded level-wise indexing of marked
	elements.
"""
function add_padding!(
    marked_elements_per_level::Vector{Vector{Int}}, space::HierarchicalFiniteElementSpace
)
    L = get_num_levels(space)
    for level in 1:L
        if marked_elements_per_level[level] == Int[]
            continue
        end

        level_space = get_space(space, level)
        basis_in_marked_elements = mapreduce(
            el -> get_basis_indices(level_space, el),
            union,
            marked_elements_per_level[level],
        )
        marked_elements_per_level[level] = mapreduce(
            basis -> get_support(level_space, basis), union, basis_in_marked_elements
        )
        intersect!(marked_elements_per_level[level], get_level_element_ids(space, level))
    end

    return marked_elements_per_level
end

"""
	get_padding_per_level(
	    space::HierarchicalFiniteElementSpace, marked_elements::Vector{Int};
	)

Separates a list of `marked_elements` in hierarchical indexing into level-wise indexing, and
adds a padding to each level.

See also [`convert_element_vector_to_elements_per_level`](@ref) and [`add_padding!`](@ref).

# Arguments
- `space::HierarchicalFiniteElementSpace`: The hierarchical finite element space.
- `marked_elements::Vector{Int};`: The list of marked elements in hierarchical indexing.

# Returns
- `element_ids_per_level::Vector{Vector{Int}}`: Level-wise indexing of marked elements.
"""
function get_padding_per_level(
    space::HierarchicalFiniteElementSpace, marked_elements::Vector{Int};
)
    element_ids_per_level = convert_element_vector_to_elements_per_level(
        space, marked_elements
    )
    add_padding!(element_ids_per_level, space)

    return element_ids_per_level
end
