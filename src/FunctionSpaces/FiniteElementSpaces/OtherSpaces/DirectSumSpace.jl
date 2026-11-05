"""
    DirectSumSpace{manifold_dim, num_components, num_patches, F} <:
       AbstractFESpace{manifold_dim, num_components, num_patches}

A multi-component space that is the direct sum of `num_components` scalar function spaces.
Consequently, their basis functions are evaluated independently. Each scalar function space
contributes to a separate component of the multi-component space, so the extraction has the
block pattern `((1, 1), ..., (num_components, num_components))` with identity blocks.

# Fields
- `component_spaces::F`: Tuple of `num_components` scalar function spaces
- `basis_offsets::NTuple{num_components, Int}`: Offsets of the basis functions of each
    component space to get the global basis functions numbers.
- `num_elements::Int`: Number of elements in the space.
- `space_dim::Int`: Dimension of the space, i.e., the number of global d.o.f.s.
"""
struct DirectSumSpace{manifold_dim, num_components, num_patches, F} <:
       AbstractFESpace{manifold_dim, num_components, num_patches}
    component_spaces::F
    basis_offsets::NTuple{num_components, Int}
    space_dim::Int

    function DirectSumSpace(
        component_spaces::F
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        F <: NTuple{num_components, AbstractFESpace{manifold_dim, 1, num_patches}},
    }
        first_space = first(component_spaces)
        foreach(space -> check_geometry(first_space, space), Base.tail(component_spaces))
        # Since each component in a direct sum space only contributes to its own component,
        # the number of global d.o.f.s is the sum of the number of d.o.f.s in each component
        # space.
        num_basis_per_component = Vector{Int}(undef, num_components)
        max_global_dof = 0
        basis_offsets = zeros(Int, num_components)
        for component_idx in 1:num_components
            num_basis_component = get_num_basis(component_spaces[component_idx])
            num_basis_per_component[component_idx] = num_basis_component

            max_global_dof += num_basis_component

            if component_idx < num_components
                basis_offsets[component_idx + 1] =
                    basis_offsets[component_idx] + num_basis_component
            end
        end

        return new{manifold_dim, num_components, num_patches, F}(
            component_spaces, NTuple{num_components, Int}(basis_offsets), max_global_dof
        )
    end
end

function get_component_spaces(space::DirectSumSpace)
    return space.component_spaces
end

function get_num_basis(space::DirectSumSpace)
    return space.space_dim
end

function get_num_basis(space::DirectSumSpace, element_id::Int)
    num_basis = 0
    for i in 1:get_num_components(space)
        num_basis += get_num_basis(get_component_spaces(space)[i], element_id)
    end
    return num_basis
end

function get_num_basis(space::DirectSumSpace, element_id::Int, component_id::Int)
    return get_num_basis(get_component_spaces(space)[component_id], element_id)
end

function get_geometry(space::DirectSumSpace)
    # The geometries of each of the component spaces are enforced to be equal, so we can
    # just pick the first.
    return get_geometry(first(get_component_spaces(space)))
end

function get_parametric_geometry(space::DirectSumSpace)
    return get_parametric_geometry(first(get_component_spaces(space)))
end

function get_basis_indices(space::DirectSumSpace, element_id::Int)
    basis_indices_per_component = ntuple(
        component_id -> get_basis_indices(space, element_id, component_id),
        get_num_components(space),
    )

    return vcat(basis_indices_per_component...)
end

function get_basis_indices(space::DirectSumSpace, element_id::Int, component_id::Int)
    return get_basis_indices(get_component_spaces(space)[component_id], element_id) .+
           get_dof_offsets(space, component_id)
end

function get_dof_offsets(space::DirectSumSpace)
    return space.basis_offsets
end

function get_dof_offsets(space::DirectSumSpace, component_id::Int)
    return space.basis_offsets[component_id]
end

function get_max_local_dim(space::DirectSumSpace)
    max_local_dim = 0
    for space in get_component_spaces(space)
        max_local_dim += get_max_local_dim(space)
    end
    return max_local_dim
end

function get_dof_partition(space::DirectSumSpace)
    component_spaces = get_component_spaces(space)
    dof_offsets = get_dof_offsets(space)

    dof_partition_per_component = FunctionSpaces.get_dof_partition.(component_spaces)
    dof_partition = deepcopy(dof_partition_per_component[1])

    for component in 2:get_num_components(space)
        for patch in eachindex(dof_partition_per_component[component])
            for partition in eachindex(dof_partition_per_component[component][patch])
                append!(
                    dof_partition[patch][partition],
                    dof_partition_per_component[component][patch][partition] .+
                    dof_offsets[component],
                )
            end
        end
    end

    return dof_partition
end

function get_local_basis(
    space::DirectSumSpace{manifold_dim, 1},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
) where {manifold_dim}
    return evaluate(only(get_component_spaces(space)), element_id, xi, nderivatives)
end

function get_block_pattern(
    ::DirectSumSpace{manifold_dim, num_components, num_patches}
) where {manifold_dim, num_components, num_patches}
    return ntuple(component_id -> (component_id, component_id), num_components)
end

function get_num_input_components(
    ::DirectSumSpace{manifold_dim, num_components, num_patches}
) where {manifold_dim, num_components, num_patches}
    return num_components
end

function get_extraction_blocks(
    ::DirectSumSpace{manifold_dim, num_components, num_patches}, element_id::Int
) where {manifold_dim, num_components, num_patches}
    return ntuple(_ -> LinearAlgebra.I, num_components)
end

function get_support(space::DirectSumSpace, basis_id::Int)
    # Find which component space the basis function belongs to.
    basis_offsets = get_dof_offsets(space)
    component_id = -1
    for c in Iterators.reverse(eachindex(basis_offsets))
        if basis_offsets[c] < basis_id
            component_id = c
            break
        end
    end
    if component_id == -1
        throw(ArgumentError(LazyString("The given basis_id ", basis_id, " was not found.")))
    end
    basis_id = basis_id - basis_offsets[component_id]

    return get_support(get_component_spaces(space)[component_id], basis_id)
end
