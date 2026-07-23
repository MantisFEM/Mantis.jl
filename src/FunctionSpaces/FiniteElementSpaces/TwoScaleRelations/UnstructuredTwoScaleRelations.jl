"""
    build_two_scale_operator(
        coarse_us_space::AbstractFESpace{manifold_dim, num_patches},
        fine_us_space::AbstractFESpace{manifold_dim, num_patches},
        nsubdivisions::NTuple{num_patches, NTuple{manifold_dim, Int}},
    ) where {manifold_dim, num_patches}

Build the two-scale operator for a general unstructured space. The fine space is assumed to
be obtained from the coarse space by subdividing the `i`-th patch into `nsubdivisions[i]`
sub-elements.

  - It is assumed that the computation of subdivision matrices for the individual function
    spaces that form the coarse unstructured space have already been implemented.
  - The global subdivision matrix for the two scale operator is computed in a brute-force
    manner by solving a least-squares problem.

# Arguments

  - `coarse_us_space::AbstractFESpace{manifold_dim,num_patches}`: The coarse unstructured space.
  - `fine_us_space::AbstractFESpace{manifold_dim,num_patches}`: The fine unstructured space.
  - `nsubdivisions::NTuple{num_patches, NTuple{manifold_dim,Int}}`: The number of subdivisions.

# Returns

  - `two_scale_op::TwoScaleOperator`: The two-scale operator.
"""
function build_two_scale_operator(
    coarse_us_space::AbstractFESpace{manifold_dim, 1, num_patches},
    fine_us_space::AbstractFESpace{manifold_dim, 1, num_patches},
    nsubdivisions::NTuple{num_patches, NTuple{manifold_dim, Int}},
) where {manifold_dim, num_patches}
    # Build the two-scale operators for the individual function spaces that form the coarse
    # unstructured space.
    discontinuous_two_scale_ops = ntuple(
        i -> build_two_scale_operator(coarse_us_space.function_spaces[i], nsubdivisions[i]),
        num_patches,
    )

    ###
    ### PART 1: Build the global subdivision matrix for the coarse and fine spaces
    ###

    # Build the global extraction operators for the coarse and fine spaces
    coarse_extraction_mat = assemble_global_extraction_matrix(coarse_us_space)
    fine_extraction_mat = assemble_global_extraction_matrix(fine_us_space)

    # Next, concatenate the two_scale_operator subdivision matrices in a block diagonal
    # format
    discontinuous_subdivision_mat = SparseArrays.blockdiag(
        [discontinuous_two_scale_ops[i][1].global_subdiv_matrix for i in 1:num_patches]...
    )

    # Finally, compute the two-scale matrix by solving a least-squares problem
    global_subdiv_matrix = SparseArrays.sparse(
        fine_extraction_mat \ Array(discontinuous_subdivision_mat * coarse_extraction_mat)
    )
    SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, global_subdiv_matrix)

    ###
    ### PART 2: Build the coarse-fine element relationships
    ###
    coarse_to_fine_elements = Vector{Vector{Int}}(undef, get_num_elements(coarse_us_space))
    # loop over coarse elements, and find global fine element ids that are contained in each
    # coarse element
    for i in 1:num_patches
        for j in 1:get_num_elements(coarse_us_space.function_spaces[i])
            global_coarse_el_id = get_global_element_id(coarse_us_space, i, j)
            local_coarse_to_fine_elements = discontinuous_two_scale_ops[i][1].coarse_to_fine_elements[j]
            coarse_to_fine_elements[global_coarse_el_id] = [
                get_global_element_id(fine_us_space, i, local_coarse_to_fine_elements[k])
                for k in eachindex(local_coarse_to_fine_elements)
            ]
        end
    end

    fine_to_coarse_elements = Vector{Int}(undef, get_num_elements(fine_us_space))
    # loop over fine elements, and find the global coarse element id that contains each
    # fine element
    for i in 1:num_patches
        for j in 1:get_num_elements(fine_us_space.function_spaces[i])
            global_fine_el_id = get_global_element_id(fine_us_space, i, j)
            fine_to_coarse_elements[global_fine_el_id] = get_global_element_id(
                coarse_us_space,
                i,
                discontinuous_two_scale_ops[i][1].fine_to_coarse_elements[j],
            )
        end
    end

    ###
    ### PART 3: Build the two-scale operator and return
    ###
    return TwoScaleOperator(
        coarse_us_space,
        fine_us_space,
        global_subdiv_matrix,
        coarse_to_fine_elements,
        fine_to_coarse_elements,
    ),
    fine_us_space
end
