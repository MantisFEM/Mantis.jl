"""
    DirectSumSpace{manifold_dim, num_components, num_patches, F} <:
       AbstractFESpace{manifold_dim, num_components, num_patches}

A multi-component space that is the direct sum of `num_components` scalar function spaces.
Consequently, their basis functions are evaluated independently and arranged in a
block-diagonal matrix. Each scalar function space contributes to a separate component of
the multi-component space.

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
    num_elements::Int
    space_dim::Int

    function DirectSumSpace(
        component_spaces::F
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        F <: NTuple{num_components, AbstractFESpace{manifold_dim, 1, num_patches}},
    }
        num_elements_per_component_space = get_num_elements.(component_spaces)
        if any(num_elements_per_component_space .!= num_elements_per_component_space[1])
            throw(
                ArgumentError("All component spaces must have the same number of elements.")
            )
        end
        # All components have the same number of elements, so we can just take the first.
        num_elements = get_num_elements(component_spaces[1])

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
            component_spaces,
            NTuple{num_components, Int}(basis_offsets),
            num_elements,
            max_global_dof,
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

function get_num_elements(space::DirectSumSpace)
    return space.num_elements
end

function get_basis_indices(space::DirectSumSpace, element_id::Int)
    basis_indices_per_component = ntuple(get_num_components(space)) do i
        return get_basis_indices(get_component_spaces(space)[i], element_id) .+
               space.basis_offsets[i]
    end

    return reduce(vcat, basis_indices_per_component)
end

function get_dof_offsets(space::DirectSumSpace)
    return space.basis_offsets
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
    space::DirectSumSpace{manifold_dim, num_components, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
    component_id::Int=1,
) where {manifold_dim, num_components, num_patches}
    evals, _ = evaluate(
        get_component_spaces(space)[component_id], element_id, xi, nderivatives
    )
    return evals
end

function get_basis_permutation(space::DirectSumSpace, element_id::Int, component_id::Int=1)
    endpoint = 0
    start = 0
    # Cumulative sum to find the correct range of basis functions for the specified
    # component. We start with computing the endpoint of the range, and then compute
    # the start from that, so that it automatically works for the first component as well
    # (where the start is 1).
    for i in 1:component_id
        # The components of the DirectSumSpace are enforced (in the constructor) to be
        # single-component spaces, so we must use `1` as the second component id here.
        lbp = length(get_basis_permutation(get_component_spaces(space)[i], element_id, 1))
        endpoint += lbp

        if i == component_id
            start = endpoint - lbp + 1
        end
    end

    return start:endpoint
end

function get_extraction(space::DirectSumSpace, element_id::Int, component_id::Int)
    return LinearAlgebra.I, get_basis_permutation(space, element_id, component_id)
end

function get_extraction_coefficients(
    space::DirectSumSpace, element_id::Int, component_id::Int
)
    return LinearAlgebra.I
end

function get_support(space::DirectSumSpace, basis_id::Int)
    # Find which component space the basis function belongs to.
    basis_offsets = get_dof_offsets(space)
    component_id = findlast(c -> c < basis_id, basis_offsets)
    basis_id = basis_id - basis_offsets[component_id]

    return get_support(get_component_spaces(space)[component_id], basis_id)
end

# function evaluate(
#     space::DirectSumSpace{manifold_dim, num_components, num_patches},
#     element_id::Int,
#     xi::Points.AbstractPoints{manifold_dim},
#     nderivatives::Int=0,
# ) where {manifold_dim, num_components, num_patches}
#     basis_indices = get_basis_indices(space, element_id)

#     # Pre-allocation, including padding (see below).
#     num_points = Points.get_num_points(xi)
#     evaluations = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
#     for j in 0:nderivatives
#         # number of derivatives of order j
#         num_j_ders = binomial(manifold_dim + j - 1, manifold_dim - 1)
#         evaluations[j + 1] = Vector{Vector{Matrix{Float64}}}(undef, num_j_ders)
#         for der_idx in 1:num_j_ders
#             evaluations[j + 1][der_idx] = [
#                 zeros(num_points, length(basis_indices)) for _ in 1:num_components
#             ]
#         end
#     end

#     # Actually evaluate the basis functions. Since DirectSumSpace is a direct sum of
#     # component spaces, we can evaluate each component space independently and then store
#     # the results in the evaluations array. We do pad with zeros to ensure a consistent and
#     # correct size of the evaluations array.
#     local_offsets = zeros(Int, num_components+1)
#     for component_idx in 1:num_components
#         # Evaluation.
#         component_eval, component_basis_idxs = evaluate(
#             get_component_spaces(space)[component_idx],
#             element_id,
#             xi,
#             nderivatives,
#         )
#         local_offsets[component_idx+1] = local_offsets[component_idx] +
#             length(component_basis_idxs)

#         for der_order in eachindex(evaluations)
#             for der_idx in eachindex(evaluations[der_order])
#                 # Padding + shift to the right position.
#                 for i in eachindex(component_basis_idxs)
#                     for point_idx in 1:num_points
#                         evaluations[der_order][der_idx][component_idx][
#                             point_idx, i + local_offsets[component_idx]
#                         ] = component_eval[der_order][der_idx][1][point_idx,i]
#                     end
#                 end
#             end
#         end
#     end

#     return evaluations, basis_indices
# end
