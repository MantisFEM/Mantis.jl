function build_two_scale_operator(
    parent_space::GTBSplineSpace{num_patches, T},
    child_space::GTBSplineSpace{num_patches, T},
    num_subdivisions::NTuple{num_patches, NTuple{1, Int}},
) where {num_patches, T}
    # Build the two-scale operators for the individual function spaces that form the parent
    # unstructured space.
    discontinuous_two_scale_ops = ntuple(
        patch -> build_two_scale_operator(
            get_patch_spaces(parent_space)[patch], num_subdivisions[patch]
        ),
        num_patches,
    )

    ###
    ### PART 1: Build the global subdivision matrix for the parent and child spaces
    ###

    # Build the global extraction operators for the parent and child spaces
    parent_extraction_mat = assemble_global_extraction_matrix(parent_space)
    child_extraction_mat = assemble_global_extraction_matrix(child_space)

    # Next, concatenate the two_scale_operator subdivision matrices in a block diagonal
    # format
    discontinuous_subdivision_mat = SparseArrays.blockdiag(
        (discontinuous_two_scale_ops[i][1].global_subdiv_matrix for i in 1:num_patches)...
    )

    # Finally, compute the two-scale matrix by solving a least-squares problem
    global_subdiv_matrix = SparseArrays.sparse(
        child_extraction_mat \ Array(discontinuous_subdivision_mat * parent_extraction_mat)
    )
    SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, global_subdiv_matrix)

    ###
    ### PART 2: Build the parent-child element relationships
    ###
    parent_to_child_elements = Vector{Vector{Int}}(undef, get_num_elements(parent_space))
    num_elements_per_patch = get_num_elements_per_patch(parent_space)
    # loop over parent elements, and find global child element ids that are contained in each
    # parent element
    for patch_id in 1:num_patches
        for loc_elem_id in 1:num_elements_per_patch[patch_id]
            global_parent_el_id = get_global_element_id(parent_space, patch_id, loc_elem_id)
            local_parent_to_child_elements = get_element_children(
                discontinuous_two_scale_ops[patch_id][1], loc_elem_id
            )
            parent_to_child_elements[global_parent_el_id] = [
                get_global_element_id(
                    child_space, patch_id, local_parent_to_child_elements[k]
                ) for k in eachindex(local_parent_to_child_elements)
            ]
        end
    end

    child_to_parent_elements = Vector{Int}(undef, get_num_elements(child_space))
    num_elements_per_patch = get_num_elements_per_patch(child_space)
    # loop over child elements, and find the global parent element id that contains each
    # child element
    for patch_id in 1:num_patches
        for loc_elem_id in 1:num_elements_per_patch[patch_id]
            global_child_el_id = get_global_element_id(child_space, patch_id, loc_elem_id)
            child_to_parent_elements[global_child_el_id] = get_global_element_id(
                parent_space,
                patch_id,
                get_element_parent(discontinuous_two_scale_ops[patch_id][1], loc_elem_id),
            )
        end
    end

    ###
    ### PART 3: Build the two-scale operator and return
    ###
    return TwoScaleOperator(
        parent_space,
        child_space,
        global_subdiv_matrix,
        parent_to_child_elements,
        child_to_parent_elements,
    ),
    child_space
end

function build_two_scale_operator(
    parent_space::GTBSplineSpace{num_patches, T},
    num_subdivisions::NTuple{num_patches, NTuple{1, Int}},
) where {num_patches, T}
    return build_two_scale_operator(
        parent_space, subdivide_space(parent_space, num_subdivisions), num_subdivisions
    )
end

function build_two_scale_operator(
    parent_space::GTBSplineSpace{num_patches, T}, num_subdivisions::NTuple{num_patches, Int}
) where {num_patches, T}
    return build_two_scale_operator(
        parent_space,
        subdivide_space(parent_space, num_subdivisions),
        ntuple(i -> (num_subdivisions[i],), num_patches),
    )
end

function build_two_scale_operator(
    parent_space::GTBSplineSpace{num_patches, T}, num_subdivisions::Int
) where {num_patches, T}
    return build_two_scale_operator(
        parent_space,
        subdivide_space(parent_space, num_subdivisions),
        ntuple(i -> (num_subdivisions,), num_patches),
    )
end

function subdivide_space(
    parent_space::GTBSplineSpace{num_patches, T},
    num_subdivisions::NTuple{num_patches, NTuple{1, Int}},
) where {num_patches, T}
    # get GTB patch spaces
    patch_spaces = get_patch_spaces(parent_space)
    # subdivide all patch spaces
    child_patch_spaces = ntuple(
        i -> subdivide_space(patch_spaces[i], num_subdivisions[i][1]), num_patches
    )
    # build child GTB spline space
    return GTBSplineSpace(
        child_patch_spaces,
        get_patch_interface_regularities(parent_space),
        get_constructor_num_dofs_left(parent_space),
        get_constructor_num_dofs_right(parent_space),
    )
end

function subdivide_space(
    parent_space::GTBSplineSpace{num_patches, T}, num_subdivisions::NTuple{num_patches, Int}
) where {num_patches, T}
    return subdivide_space(parent_space, ntuple(i -> (num_subdivisions[i],), num_patches))
end

function subdivide_space(
    parent_space::GTBSplineSpace{num_patches, T}, num_subdivisions::Int
) where {num_patches, T}
    return subdivide_space(parent_space, ntuple(i -> (num_subdivisions,), num_patches))
end
