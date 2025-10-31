"""
    GTBSplineSpace{num_patches, T, TE, TI, TJ} <: AbstractFESpace{1, 1, num_patches}

Constructs a GTBSplineSpace from a tuple of NURBS or B-spline spaces and a vector of
regularity conditions. Periodic spaces are a special case of GTBSplines for
`num_patches = 1` and `regularity = [r]` with `r > -1`.

# Fields
- `patch_spaces::NTuple{num_patches, T}`: A tuple of `num_patches` NURBS or B-spline spaces.
- `extraction_op::ExtractionOperator{1, TE, TI, TJ}`: The extraction operator for the
    GTBSpline space.
- `dof_partition::Vector{Vector{Vector{Int}}}`: A vector of vectors of vectors of integers
    representing the degree of freedom partitioning.
- `regularity::Vector{Int}`: A vector of regularity conditions at the interfaces between
    the spaces.

# Arguments for constructor
- `patch_spaces::NTuple{num_patches, T}`: A tuple of `num_patches` NURBS or B-spline spaces.
- `regularity::Vector{Int}`: A vector of regularity conditions at the interfaces between
    the spaces.
- `num_dofs_left::Int`: The number of degrees of freedom at the left boundary. Default is
    -1, which means it will be computed automatically.
- `num_dofs_right::Int`: The number of degrees of freedom at the right boundary. Default is
    -1, which means it will be computed automatically.

# Throws
- `ArgumentError`: If the number of regularity conditions does not match the number of
    interfaces.
- `ArgumentError`: If the minimal polynomial degree of any pair of adjacent spaces is less
    than the corresponding regularity condition.
"""
struct GTBSplineSpace{num_patches, T, TE, TI, TJ} <: AbstractFESpace{1, 1, num_patches}
    patch_spaces::T
    extraction_op::ExtractionOperator{1, TE, TI, TJ}
    dof_partition::Vector{Vector{Vector{Int}}}
    num_elements_per_patch::NTuple{num_patches, Int}
    regularity::Vector{Int}
    num_dofs_left::Int
    num_dofs_right::Int

    function GTBSplineSpace(
        patch_spaces::T,
        regularity::Vector{Int},
        num_dofs_left::Int=-1,
        num_dofs_right::Int=-1,
    ) where {
        num_patches, S <: Union{BSplineSpace, RationalFESpace}, T <: NTuple{num_patches, S}
    }
        # Check if the number of regularity conditions matches the number of interfaces
        if length(regularity) != num_patches
            throw(ArgumentError("""\
                The number of regularity conditions should be equal to the number of \
                bspline interfaces. You have $(num_patches) interfaces and \
                $(length(regularity)) regularity conditions.\
                """))
        end

        # Check if the polynomial degree of each pair of adjacent spaces is sufficient for
        # the regularity condition
        for i in 1:num_patches
            j = i
            k = i + 1
            if i == num_patches
                k = 1
            end
            polynomial_degree = min(
                get_polynomial_degree(patch_spaces[j], 1),
                get_polynomial_degree(patch_spaces[k], 1),
            )
            if polynomial_degree < regularity[i]
                throw(ArgumentError("""\
                    The minimal polynomial degrees must be greater than or equal to the \
                    regularity. The minimal degree is $(polynomial_degree) and there is
                    regularity $(regularity[i]) at index $(i).\
                    """))
            end
        end

        # Create the extraction operator
        extraction_op = extract_gtbspline_to_bspline(patch_spaces, regularity)
        # number of element offsets per patch
        num_elements_per_patch = ntuple(num_patches) do i
            return get_num_elements(patch_spaces[i])
        end
        num_elements_offset = cumsum([0; collect(get_num_elements.(patch_spaces))])

        function _get_boundary_dof_inds(patch_id::Int, right_bnd::Bool)
            # get dof partitioning of the first patch space
            dof_partition_loc = get_dof_partition(patch_spaces[patch_id])
            # get all non-zero basis function indices at boundary
            patch_el_id = 1
            bnd_id = 1
            if right_bnd
                patch_el_id = get_num_elements(patch_spaces[patch_id])
                bnd_id = 3
            end
            # get indices as per local numbering
            dof_inds_loc = get_basis_indices(patch_spaces[patch_id], patch_el_id)
            nnz_loc = [findfirst(dof_inds_loc .== i) for i in dof_partition_loc[1][bnd_id]]
            # find all non-zero basis function indices at left end of first element
            el_id = num_elements_offset[patch_id] + 1
            if right_bnd
                el_id = num_elements_offset[patch_id + 1]
            end
            dof_inds_global = get_basis_indices(extraction_op, el_id)
            nnz_global = findall(
                get_extraction_coefficients(extraction_op, el_id)[nnz_loc, :] .!= 0
            )
            return unique([
                dof_inds_global[nnz_global[i][2]] for i in eachindex(nnz_global)
            ])
        end

        # Allocate memory for degree of freedom partitioning: note, even if the 1D space is
        # implemented as a multipatch object, the dof partition treats it as a single patch!
        dof_partition = Vector{Vector{Vector{Int}}}(undef, num_patches)
        for i in 1:num_patches
            dof_partition[i] = Vector{Vector{Int}}(undef, 3)
        end
        # Construct the dof-partitioning at the beginning and end of the patch.
        if num_dofs_left >= 0 # the number of dofs has been manually set
            dof_partition[1][1] = collect(1:num_dofs_left)
        else
            dof_partition[1][1] = _get_boundary_dof_inds(1, false)
        end
        if num_dofs_right >= 0 # the number of dofs has been manually set
            dof_partition[num_patches][3] = collect(
                (get_num_basis(extraction_op) - num_dofs_right + 1):get_num_basis(
                    extraction_op
                ),
            )
        else
            dof_partition[num_patches][3] = _get_boundary_dof_inds(num_patches, true)
        end

        # Then build the dof-partitioning at the interior patch interfaces ...
        for i in 2:(num_patches)
            dof_partition[i - 1][3] = _get_boundary_dof_inds(i - 1, true)
            dof_partition[i][1] = _get_boundary_dof_inds(i, false)
        end
        # ... and then the interiors.
        for i in 1:num_patches
            nel1 = num_elements_offset[i]
            nel2 = num_elements_offset[i + 1]
            bnd_dofs = vcat(dof_partition[i][1], dof_partition[i][3])
            dof_partition[i][2] = setdiff(
                get_basis_indices(extraction_op, nel1 + 1), bnd_dofs
            )
            for j in (nel1 + 1):nel2
                append!(
                    dof_partition[i][2],
                    setdiff(get_basis_indices(extraction_op, j), bnd_dofs),
                )
            end
            dof_partition[i][2] = unique(dof_partition[i][2])
        end

        return new{num_patches, T, get_EIJ_types(extraction_op)...}(
            patch_spaces,
            extraction_op,
            dof_partition,
            num_elements_per_patch,
            regularity,
            num_dofs_left,
            num_dofs_right
        )
    end
end

function get_num_elements_per_patch(space::GTBSplineSpace)
    return space.num_elements_per_patch
end

function get_num_local_basis_per_patch(
    space::GTBSplineSpace{num_patches}
) where {num_patches}
    return ntuple(num_patches) do i
        return get_num_basis(get_patch_spaces(space)[i])
    end
end

function get_element_lengths(space::GTBSplineSpace, element_id::Int)
    patch_id, local_element_id = get_patch_and_local_element_id(space, element_id)

    return get_element_lengths(get_patch_spaces(space)[patch_id], local_element_id)
end

function get_local_basis(
    space::GTBSplineSpace,
    element_id::Int,
    xi::Points.AbstractPoints{1},
    nderivatives::Int,
    component_id::Int=1,
)
    # We need space ID and local element ID to know which function space to evaluate.
    patch_id, patch_element_id = get_patch_and_local_element_id(space, element_id)

    return evaluate(get_patch_spaces(space)[patch_id], patch_element_id, xi, nderivatives)[1]
end

get_patch_spaces(space::GTBSplineSpace) = space.patch_spaces
function get_max_local_dim(space::GTBSplineSpace)
    return maximum(get_max_local_dim.(get_patch_spaces(space)))
end

function get_derivative_space(space::GTBSplineSpace{num_patches}) where {num_patches}
    patch_spaces_der = ntuple(num_patches) do i
        return get_derivative_space(get_patch_spaces(space)[i])
    end
    regularity_der = space.regularity .- 1
    regularity_der = [max(r, -1) for r in regularity_der]

    return GTBSplineSpace(
        patch_spaces_der,
        regularity_der,
        max(space.num_dofs_left-1, -1),
        max(space.num_dofs_right-1, -1),
    )
end

function assemble_global_extraction_matrix(
    space::GTBSplineSpace{num_patches}
) where {num_patches}
    # Number of global basis functions
    num_global_basis = get_num_basis(space)
    # Number of elements
    nel = get_num_elements(space)
    # Number of local basis functions
    num_local_basis = get_num_local_basis_per_patch(space)
    num_local_basis_offset = cumsum([0; num_local_basis...])
    # Initialize the global extraction matrix
    global_extraction_matrix = zeros(Float64, num_local_basis_offset[end], num_global_basis)

    # Loop over all patches
    for el_id in 1:nel
        patch_id, local_element_id = get_patch_and_local_element_id(space, el_id)
        # get extraction coefficients on this element
        extraction_coefficients = get_extraction_coefficients(space, el_id)
        global_basis_indices = get_basis_indices(space, el_id)
        # get local basis indices
        local_basis_indices =
            get_basis_indices(get_patch_spaces(space)[patch_id], local_element_id) .+
            num_local_basis_offset[patch_id]

        # Assemble the global extraction matrix
        global_extraction_matrix[local_basis_indices, global_basis_indices] =
            extraction_coefficients
    end

    return SparseArrays.sparse(global_extraction_matrix)
end

get_constructor_num_dofs_left(space::GTBSplineSpace) = space.num_dofs_left
get_constructor_num_dofs_right(space::GTBSplineSpace) = space.num_dofs_right
get_patch_interface_regularities(space::GTBSplineSpace) = space.regularity
