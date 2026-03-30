struct C0Space{manifold_dim, num_patches, T, G, GP, TE, TI, TJ} <: AbstractFESpace{manifold_dim, 1, num_patches}
    function_spaces::T
    geometry::G
    parametric_geometry::GP
    extraction_op::ExtractionOperator{1, TE, TI, TJ}
    dof_partition::Vector{Vector{Vector{Int}}}
    global_to_local_dof_dict::Dict{Int, Dict{Int, Int}}
    local_to_global_dof_dict::Dict{Tuple{Int, Int}, Int}
end

function C0Space(
    function_spaces::T, geometry::G
) where {manifold_dim, image_dim, num_patches, T <: NTuple{num_patches, AbstractFESpace{manifold_dim, 1, 1}}, G <: Geometry.AbstractGeometry{manifold_dim, image_dim, num_patches}}
    # Compute the total number of elements and the offsets for each patch.
    num_elements = 0
    num_elements_per_patch = map(get_num_elements, function_spaces)
    num_elements = sum(num_elements_per_patch)
    elems_per_patch_offset = vcat(0, [(cumsum(num_elements_per_patch[1:(end - 1)]))...])

    # Create the dof partition, accounting for shared dofs.
    global_dof = 0
    dof_partition = Vector{Vector{Vector{Int}}}(undef, num_patches)
    global_to_local_dof_dict = Dict{Int, Dict{Int, Int}}()
    local_to_global_dof_dict = Dict{Tuple{Int, Int}, Int}()
    num_divisions = 3^manifold_dim
    # First number all interior dofs, these are never shared so this is just a matter of
    # assigning a number. These are always the manifold_dim sized containers, that is,
    # surfaces in 2D, volumes in 3D, etc.
    interior_division = Topology.id_to_dof_division(manifold_dim, manifold_dim, 1)
    for (patch_id, space) in pairs(function_spaces)
        dof_partition[patch_id] = Vector{Vector{Int}}(undef, num_divisions)

        local_interior_dofs = get_interior_dofs(space, 1)
        num_interior_dofs = length(local_interior_dofs)

        global_interior_dofs = (global_dof+1):(global_dof+num_interior_dofs)
        dof_partition[patch_id][interior_division] = global_interior_dofs

        for (local_dof, global_interior_dof) in zip(local_interior_dofs, global_interior_dofs)
            local_to_global_dof_dict[(patch_id, local_dof)] = global_interior_dof
            global_to_local_dof_dict[global_interior_dof] = Dict{Int, Int}(
                patch_id => local_dof
            )
        end

        global_dof += num_interior_dofs
    end
    # Makes sure we can assign the dofs by initialising the arrays.
    for i in eachindex(dof_partition)
        for j in eachindex(dof_partition[i])
           if j != interior_division
              dof_partition[i][j] = Int[]
           end
        end
    end
    # Then we number all bounding entities. So all topological objects of dimension
    # manifold_dim-1 or less.
    topology = Geometry.get_topology(geometry)
    boundaries, interfaces = Topology.get_boundaries_and_interfaces(topology)
    # We start with the boundaries (these are not shared).
    for (dim, boundary_id) in boundaries
        patch_id = topology[dim+1, manifold_dim+1][boundary_id][1] # There is only one patch.
        local_boundary_id = abs(Topology.get_local_id(topology, patch_id, boundary_id, dim))
        boundary_dof_division = Topology.id_to_dof_division(manifold_dim, dim, local_boundary_id)
        for local_dof in get_dofs(function_spaces[patch_id], 1, dim, local_boundary_id)
            global_dof += 1
            push!(dof_partition[patch_id][boundary_dof_division], global_dof)
            local_to_global_dof_dict[(patch_id, local_dof)] = global_dof
            global_to_local_dof_dict[global_dof] = Dict{Int, Int}(patch_id => local_dof)
        end
    end
    # Finally, we number all interface dofs, these are shared.
    for (dim, interface_id) in interfaces
        patch_ids = topology[dim+1, manifold_dim+1][interface_id]

        # Process the first patch in the list, here we assign the global dofs.
        patch_id = patch_ids[1]
        local_interface_id = abs(Topology.get_local_id(topology, patch_id, interface_id, dim))
        first_interface_dof_division = Topology.id_to_dof_division(manifold_dim, dim, local_interface_id)
        dofs_patch_1 = get_dofs(function_spaces[patch_id], 1, dim, local_interface_id)
        for local_dof in dofs_patch_1
            global_dof += 1
            push!(dof_partition[patch_id][first_interface_dof_division], global_dof)
            local_to_global_dof_dict[(patch_id, local_dof)] = global_dof
            global_to_local_dof_dict[global_dof] = Dict{Int, Int}(patch_id => local_dof)
        end

        # Save the current patch_id and its number of dofs for checking.
        patch1 = patch_id
        local_interface_1 = local_interface_id
        num_dofs_at_interface_patch_1 = length(dofs_patch_1)

        # Process the remaining patches. We already assigned to global dofs, so only need
        # to obtain the correspondence to the local dofs.
        for patch_id in patch_ids[2:end]
            local_interface_id = abs(Topology.get_local_id(topology, patch_id, interface_id, dim))
            interface_dof_division = Topology.id_to_dof_division(manifold_dim, dim, local_interface_id)

            dofs_patch_i = get_dofs(function_spaces[patch_id], 1, dim, local_interface_id)

            # Check that the number of dofs on all sides of the interface match.
            num_dofs_at_interface_patch_i = length(dofs_patch_i)
            if num_dofs_at_interface_patch_i != num_dofs_at_interface_patch_1
                throw(
                    ArgumentError(
                        LazyString(
                            "While processing interface ",
                            interface_id,
                            " of dimension ",
                            dim,
                            " (0 = point, 1 = edge, 2 = surface, etc.) with local ids",
                            " (patch_id, local_interface_id) ",
                            (patch1, local_interface_1),
                            " and ",
                            (patch_id, local_interface_id),
                            ", the space on patch ",
                            patch1,
                            " was found to have ",
                            num_dofs_at_interface_patch_1,
                            " dofs on this interface, while the space on patch ",
                            patch_id,
                            " has ",
                            num_dofs_at_interface_patch_i,
                            " dofs on this interface. The spaces can't be stitched",
                            " together if the number of dofs on an interface doesn't match.",
                        )
                    )
                )
            end
            for (local_dof, global_dof) in zip(
                dofs_patch_i,
                dof_partition[patch_ids[1]][first_interface_dof_division]
            )
                push!(dof_partition[patch_id][interface_dof_division], global_dof)
                local_to_global_dof_dict[(patch_id, local_dof)] = global_dof
                global_to_local_dof_dict[global_dof][patch_id] = local_dof
            end
        end
    end

    # Create the global extraction operator.
    extraction_coefficients = Vector{NTuple{1, typeof(LinearAlgebra.I)}}(
        undef, num_elements
    )
    basis_indices = Vector{Indices{1, Vector{Int}, UnitRange{Int}}}(undef, num_elements)
    for patch_id in 1:num_patches
        for local_elem_id in 1:num_elements_per_patch[patch_id]
            global_elem_id = elems_per_patch_offset[patch_id] + local_elem_id

            # Get the local extraction coefficients and basis indices.
            indices = get_basis_indices(function_spaces[patch_id], local_elem_id)

            extraction_coefficients[global_elem_id] = (LinearAlgebra.I,)

            basis_indices[global_elem_id] = Indices(
                [local_to_global_dof_dict[(patch_id, local_dof)] for local_dof in indices],
                (1:length(indices),),
            )
        end
    end

    E = ExtractionOperator(extraction_coefficients, basis_indices, num_elements, global_dof)

    parametric_geometry = Geometry.UnstructuredGeometry(map(get_geometry, function_spaces))

    return C0Space{manifold_dim, num_patches, T, G, typeof(parametric_geometry), get_EIJ_types(E)...}(
        function_spaces,
        geometry,
        parametric_geometry,
        E,
        dof_partition,
        global_to_local_dof_dict,
        local_to_global_dof_dict,
    )
end

get_patch_spaces(space::C0Space) = space.function_spaces

function get_local_basis(
    space::C0Space{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
    component_id::Int=1,
) where {manifold_dim}
    patch_id, local_element_id = get_patch_and_local_element_id(space, element_id)

    # Only keep the evaluations, not the indices.
    return evaluate(space.function_spaces[patch_id], local_element_id, xi, nderivatives)[1]
end

function get_num_elements_per_patch(space::C0Space)
    return map(get_num_elements, get_patch_spaces(space))
end

function get_max_local_dim(space::C0Space)
    max_local_dim = mapreduce(get_max_local_dim, max, get_patch_spaces(space))
    return max_local_dim
end

function get_element_lengths(space::C0Space, element_id::Int)
    patch_id, local_element_id = get_patch_and_local_element_id(space, element_id)
    return get_element_lengths(space.function_spaces[patch_id], local_element_id)
end

function get_element_vertices(space::C0Space, element_id::Int)
    patch_id, local_element_id = get_patch_and_local_element_id(space, element_id)
    return get_element_vertices(space.function_spaces[patch_id], local_element_id)
end

function get_support(space::C0Space, basis_id::Int)
    local_basis_ids = space.global_to_local_dof_dict[basis_id]
    @show local_basis_ids
    support = Int[]
    for (patch_id, local_basis_id) in local_basis_ids
        local_support = get_support(space.function_spaces[patch_id], local_basis_id)
        global_support = local_support
        for i in 1:(patch_id - 1)
            global_support .+= get_num_elements(space.function_spaces[i])
        end
        append!(support, global_support)
    end
    return support
end
