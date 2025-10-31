function build_two_scale_operator(
    parent_space::PolarSplineSpace,
    child_space::PolarSplineSpace,
    num_subdivisions::NTuple{1, NTuple{2, Int}},
)
    # Build the two-scale operators for the individual function spaces that form the parent
    # unstructured space.
    discontinuous_two_scale_op, _ = build_two_scale_operator(
        DirectSumSpace(get_patch_spaces(parent_space)), num_subdivisions[1]
    )

    ###
    ### PART 1: Build the global subdivision matrix for the parent and child spaces
    ###

    # Build the global extraction operators for the parent and child spaces
    parent_extraction_mat = assemble_global_extraction_matrix(parent_space)
    child_extraction_mat = assemble_global_extraction_matrix(child_space)

    # Next, concatenate the two_scale_operator subdivision matrices in a block diagonal
    # format
    discontinuous_subdivision_mat = get_global_subdiv_matrix(discontinuous_two_scale_op)

    # Finally, compute the two-scale matrix by solving a least-squares problem
    global_subdiv_matrix = SparseArrays.sparse(
        child_extraction_mat \ Array(discontinuous_subdivision_mat * parent_extraction_mat)
    )
    SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, global_subdiv_matrix)

    ###
    ### PART 2: Build the two-scale operator and return
    ###
    return TwoScaleOperator(
        parent_space,
        child_space,
        global_subdiv_matrix,
        get_parent_to_children_elements(discontinuous_two_scale_op),
        get_child_to_parent_elements(discontinuous_two_scale_op),
    ),
    child_space
end

function build_two_scale_operator(
    parent_space::PolarSplineSpace,
    num_subdivisions::NTuple{1, NTuple{2, Int}},
)
    return build_two_scale_operator(
        parent_space, subdivide_space(parent_space, num_subdivisions), num_subdivisions
    )
end

function build_two_scale_operator(
    parent_space::PolarSplineSpace,
    num_subdivisions::NTuple{2, Int},
)
    return build_two_scale_operator(parent_space, (num_subdivisions,))
end

function build_two_scale_operator(
    parent_space::PolarSplineSpace, num_subdivisions::Int
)
    return build_two_scale_operator(
        parent_space, (num_subdivisions, num_subdivisions)
    )
end

function subdivide_space(
    parent_space::PolarSplineSpace,
    num_subdivisions::NTuple{1, NTuple{2, Int}},
)

    ############################################
    # Refine degenerate mapping data
    ############################################

    # get coarse degenerate mapping coefficients
    degen_cp_parent = get_degenerate_control_points(parent_space)
    size_degen_cp_parent = size(degen_cp_parent)
    # refine degenerate mapping
    TS_tp, degen_space_child = build_two_scale_operator(
        get_degenerate_space(parent_space),
        num_subdivisions[1]
    )
    size_tp_child = FunctionSpaces.get_num_basis.(
        FunctionSpaces.get_constituent_spaces(degen_space_child)
    )
    subdiv_mat_tp = FunctionSpaces.get_global_subdiv_matrix(TS_tp)
    degen_cp_child = reshape(
        subdiv_mat_tp * reshape(degen_cp_parent, :, size_degen_cp_parent[3]),
        (size_tp_child[1], size_tp_child[2], size_degen_cp_parent[3]),
    )

    ############################################
    # Build finer patch spaces
    ############################################

    patch_spaces_child = subdivide_space.(
        get_patch_spaces(parent_space),
        Ref(num_subdivisions[1]),
    )

    ############################################
    # Build finer polar space
    ############################################
    return PolarSplineSpace(
        patch_spaces_child,
        degen_cp_child,
        degen_space_child,
        parent_space.two_poles,
        parent_space.zero_at_poles,
    )
end

function subdivide_space(
    parent_space::PolarSplineSpace,
    num_subdivisions::NTuple{2, Int},
)
    return subdivide_space(parent_space, (num_subdivisions,))
end

function subdivide_space(parent_space::PolarSplineSpace, num_subdivisions::Int)
    return subdivide_space(parent_space, (num_subdivisions, num_subdivisions))
end
