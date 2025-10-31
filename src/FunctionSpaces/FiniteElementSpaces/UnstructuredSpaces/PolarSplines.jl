############################################################################################
# General methods
############################################################################################

"""
    _build_polar_extraction_and_dof_partition(
        tp_space::TensorProductSpace,
        E::SparseMatrixCSC{Float64,Int},
        two_poles::Bool
    )

Build the extraction operator and dof partitioning for the polar spline space.

# Arguments
- `tp_space::TensorProductSpace`: The tensor product space.
- `E::SparseMatrixCSC{Float64,Int}`: The extraction operator.
- `two_poles::Bool`: Whether the polar spline space has two poles.

# Returns
- `::Tuple{ExtractionOperator, Vector{Vector{Vector{Int}}}}`: The extraction operator and dof partitioning.
"""
function _build_polar_extraction_and_dof_partition(
    tp_space::NTuple{num_components, TP}, E::NTuple{num_components, TE}, two_poles::Bool
) where {num_components, TP, TE}
    # number of elements
    # assumption: all component spaces have the same number of elements
    num_elements = get_num_elements(tp_space[1])

    # allocate space for extraction coefficients and basis indices
    extraction_coefficients = Vector{NTuple{num_components, Matrix{Float64}}}(
        undef, num_elements
    )
    basis_indices = Vector{Indices{num_components, Vector{Int}, Vector{Int}}}(
        undef, num_elements
    )

    # Convert global extraction matrix to element local extractions
    # (the matrix is transposed so that [tp_spline_space] * [extraction] = [Polar-splines])
    cols_el = Vector{Vector{Int}}(undef, num_components)
    rows_el = Vector{Vector{Int}}(undef, num_components)
    for el_idx in 1:num_elements
        for component_idx in 1:num_components
            # get the indices of the component bases which contribute to the global
            # multicomponent polar bases
            cols_el[component_idx] = get_basis_indices(tp_space[component_idx], el_idx)
            # get the indices of the global multicomponent polar bases
            rows_el[component_idx] = unique(
                SparseArrays.findnz(E[component_idx][:, cols_el[component_idx]])[1]
            )
        end
        # unique indices for global multicomponent polar bases
        basis_indices_el = unique(cat(rows_el...; dims=1))
        # permutation mapping on element
        index_perm_el = Vector{Vector{Int}}(undef, num_components)
        # Matrix of coefficients on element
        coeffs_el = Vector{Matrix{Float64}}(undef, num_components)
        for component_idx in 1:num_components
            index_perm_el[component_idx] = findall(
                x -> x in rows_el[component_idx], basis_indices_el
            )
            coeffs_el[component_idx] = permutedims(
                Array(
                    E[component_idx][
                        basis_indices_el[index_perm_el[component_idx]],
                        cols_el[component_idx],
                    ],
                ),
            )
        end

        # store basis indices and permutation on element
        basis_indices[el_idx] = Indices(basis_indices_el, (index_perm_el...,))
        # store Matrix of coefficients
        extraction_coefficients[el_idx] = (coeffs_el...,)
    end

    # number of polar dofs
    # assume all global component matrices E[i] have the same number of rows
    space_dim = size(E[1], 1)

    # Polar spline extraction operator
    extraction_op = ExtractionOperator(
        extraction_coefficients, basis_indices, num_elements, space_dim
    )

    # TODO: the following only uses the partitioning of the first component tp_space, this
    # is wrong for multicomponent spaces
    dof_partition = Vector{Vector{Vector{Int}}}(undef, 1)
    # dof partitioning for the tensor product space
    dof_partition_tp = get_dof_partition(tp_space[1])
    n_partn = length(dof_partition_tp[1])
    dof_partition[1] = Vector{Vector{Int}}(undef, n_partn)
    if two_poles
        for i in 1:n_partn
            dof_partition[1][i] = []
        end
    else
        for i in 1:n_partn
            if length(dof_partition_tp[1][i]) == 0
                dof_partition[1][i] = []
            else
                # TODO!!
                dof_partition[1][i] = []
            end
        end
    end

    # return data needed for building polar splines
    return extraction_op, dof_partition
end

"""
    _get_circumscribing_triangle(points, origin)

Build a standard triangle that contains the input points.

# Arguments
- `points::Matrix{Float64}`: The input points, where each row corresponds to a point.
- `origin::Vector{Float64}`: The point which is translated to the origin.

# Returns
- `::Matrix{Float64}`: The circumscribing triangle.
"""
function _get_circumscribing_triangle(
    points::Matrix{Float64}, origin::Vector{Float64}=[0.0, 0.0]
)
    rad = sqrt.(sum((points .- permutedims(origin)) .^ 2; dims=2))
    tau = 2 * maximum(rad)
    trix = [tau, -tau / 2, -tau / 2]
    triy = [0, -sqrt(3) * tau / 2, sqrt(3) * tau / 2]
    return [trix triy]
end

"""
    _get_barycentric_coordinates(points, tri, origin)

Compute the barycentric coordinates of the input points w.r.t. the control triangle. The
points are translated by the offset before computing the barycentric coordinates.

# Arguments
- `points::Matrix{Float64}`: The input points, where each row corresponds to a point.
- `tri::Matrix{Float64}`: Matrix with the coordinates of the `i`-th vertex of the control
    triangle in the `i`-th row.
- `origin::Vector{Float64}`: The point which is translated to the origin.

# Returns
- `::Matrix{Float64}`: The barycentric coordinates.
"""
function _get_barycentric_coordinates(
    points::Matrix{Float64}, tri::Matrix{Float64}, origin::Vector{Float64}=[0.0, 0.0]
)
    L = [tri ones(3)]
    return hcat(points .- permutedims(origin), ones(size(points, 1))) / L
end

############################################################################################
# PolarSplines
############################################################################################

struct PolarSplineSpace{num_components, T, TD, TE, TI, TJ} <:
       AbstractFESpace{2, num_components, 1}
    patch_spaces::T
    extraction_op::ExtractionOperator{num_components, TE, TI, TJ}
    dof_partition::Vector{Vector{Vector{Int}}}
    num_elements_per_patch::NTuple{1, Int}
    regularity::Int

    global_extraction_matrix::NTuple{
        num_components, SparseArrays.SparseMatrixCSC{Float64, Int}
    }
    control_triangle::Matrix{Float64}
    two_poles::Bool
    zero_at_poles::Bool
    degenerate_control_points::Array{Float64, 3}
    degenerate_space::TD

    """
        PolarSplineSpace(
            patch_spaces::NTuple{1, TensorProductSpace{2, 1}},
            degenerate_control_points::Array{Float64, 3},
            degenerate_space::TensorProductSpace{2, 1},
            two_poles::Bool=false,
            zero_at_poles::Bool=false,
        )

    Build a scalar polar spline space from the given tensor-product space and degenerate
    control points and the corresponding degenerate tensor-product space (the latter two
    define the polar parametric domain).

    # Arguments
    - `patch_spaces::NTuple{1, TensorProductSpace{2, 1}}`: The tensor-product space defining
        the polar spline space.
    - `degenerate_control_points::Array{Float64, 3}`: The degenerate control points.
    - `degenerate_space::TensorProductSpace{2, 1}`: The degenerate tensor-product space.
    - `two_poles::Bool=false`: Whether the polar spline space has two poles.
    - `zero_at_poles::Bool=false`: Whether functions are constrained to zero at poles.

    # Returns
    - `::PolarSplineSpace{1, typeof(patch_spaces), typeof(degenerate_space), TE..., TI..., TJ...}`:
        The scalar polar spline space.
    """
    function PolarSplineSpace(
        patch_spaces::NTuple{1, TensorProductSpace{2, 1}},
        degenerate_control_points::Array{Float64, 3},
        degenerate_space::TensorProductSpace{2, 1},
        two_poles::Bool=false,
        zero_at_poles::Bool=false,
    )
        # poloidal and radial spaces
        space_p, space_r = get_constituent_spaces(patch_spaces[1])
        # number of basis functions for the poloidal and radial spaces
        n_p = get_num_basis(space_p)
        n_r = get_num_basis(space_r) + zero_at_poles
        if n_p != size(degenerate_control_points, 1)
            throw(
                ArgumentError(
                    "The poloidal space does not match the input coefficients."
                ),
            )
        end
        if n_r != size(degenerate_control_points, 2)
            throw(
                ArgumentError(
                    "The radial space does not match the input coefficients."
                ),
            )
        end
        if n_p != get_num_basis(get_constituent_spaces(degenerate_space)[1])
            throw(
                ArgumentError(
                    "The degenerate tensor-product space does not match the input coefficients.",
                ),
            )
        end
        if n_r != get_num_basis(get_constituent_spaces(degenerate_space)[2])
            throw(
                ArgumentError(
                    "The degenerate tensor-product space does not match the input coefficients.",
                ),
            )
        end

        # first, build extraction operator and control triangle
        E, control_triangle = extract_scalar_polar_splines_to_tensorproduct(
            degenerate_control_points, n_r, two_poles, zero_at_poles
        )

        # zero_at_poles = false: C^1 smooth scalars
        # zero_at_poles = true: C^-1 smooth scalars
        regularity = 1
        if zero_at_poles
            regularity = -1
        end

        # build polar spline extraction operator and dof partitioning
        extraction_op, dof_partition = _build_polar_extraction_and_dof_partition(
            patch_spaces, (E,), two_poles
        )

        return new{
            1,
            typeof(patch_spaces),
            typeof(degenerate_space),
            get_EIJ_types(extraction_op)...
        }(
            patch_spaces,
            extraction_op,
            dof_partition,
            (get_num_elements(patch_spaces[1]),),
            regularity,
            (E,),
            control_triangle,
            two_poles,
            zero_at_poles,
            degenerate_control_points,
            degenerate_space
        )
    end

    """
        PolarSplineSpace(
            patch_spaces::NTuple{2, TensorProductSpace{2, 1}},
            degenerate_control_points::Array{Float64, 3},
            degenerate_space::TensorProductSpace{2, 1},
            two_poles::Bool=false,
            ::Bool=false
        )

    Build a vector polar spline space from the given tensor-product spaces and degenerate
    control points and the corresponding degenerate tensor-product space (the latter two
    define the polar parametric domain).

    # Arguments
    - `patch_spaces::NTuple{2, TensorProductSpace{2, 1}}`: The tensor-product spaces defining
        the polar spline space.
    - `degenerate_control_points::Array{Float64, 3}`: The degenerate control points.
    - `degenerate_space::TensorProductSpace{2, 1}`: The degenerate tensor-product space.
    - `two_poles::Bool=false`: Whether the polar spline space has two poles.
    - `::Bool=false`: Dummy argument to have a uniform constructor as the scalar case.

    # Returns
    - `::PolarSplineSpace{2, typeof(patch_spaces), typeof(degenerate_space), TE..., TI..., TJ...}`:
        The vector polar spline space.
    """
    function PolarSplineSpace(
        patch_spaces::NTuple{2, TensorProductSpace{2, 1}},
        degenerate_control_points::Array{Float64, 3},
        degenerate_space::TensorProductSpace{2, 1},
        two_poles::Bool=false,
        ::Bool=false,
    )
        # poloidal and radial component spaces
        dspace_p, space_r = get_constituent_spaces(patch_spaces[1])
        space_p, dspace_r = get_constituent_spaces(patch_spaces[2])
        # number of basis functions for the poloidal and radial spaces
        n_p = get_num_basis(space_p)
        n_r = get_num_basis(space_r)
        if n_p != size(degenerate_control_points, 1)
            throw(
                ArgumentError(
                    "The poloidal space does not match the input coefficients."
                ),
            )
        end
        if n_r != size(degenerate_control_points, 2)
            throw(
                ArgumentError(
                    "The radial space does not match the input coefficients."
                ),
            )
        end
        if n_p != get_num_basis(get_constituent_spaces(degenerate_space)[1])
            throw(
                ArgumentError(
                    "The degenerate tensor-product space does not match the input coefficients.",
                ),
            )
        end
        if n_r != get_num_basis(get_constituent_spaces(degenerate_space)[2])
            throw(
                ArgumentError(
                    "The degenerate tensor-product space does not match the input coefficients.",
                ),
            )
        end
        if n_p != get_num_basis(dspace_p)
            throw(
                ArgumentError(
                    "Input poloidal space incompatible with periodicity assumption."
                ),
            )
        end
        if n_r != get_num_basis(dspace_r) + 1
            throw(
                ArgumentError(
                    "Input radial space and its derivative are incompatible."
                ),
            )
        end

        # first, build extraction operator and control triangle
        E, control_triangle = extract_vector_polar_splines_to_tensorproduct(
            degenerate_control_points, n_r, two_poles
        )

        # build polar spline extraction operator and dof partitioning
        extraction_op, dof_partition = _build_polar_extraction_and_dof_partition(
            patch_spaces, E, two_poles
        )

        regularity = 0
        return new{
            2,
            typeof(patch_spaces),
            typeof(degenerate_space),
            get_EIJ_types(extraction_op)...
        }(
            patch_spaces,
            extraction_op,
            dof_partition,
            (get_num_elements(patch_spaces[1]),),
            regularity,
            E,
            control_triangle,
            two_poles,
            false,
            degenerate_control_points,
            degenerate_space
        )
    end
end

"""
    extract_scalar_polar_splines_to_tensorproduct(
        degenerate_control_points::Array{Float64, 3},
        num_basis_r::Int,
        two_poles::Bool=false,
        zero_at_poles::Bool=nothing
    )

Build the extraction operator to extract the polar spline basis functions from the tensor
product space.

# Arguments
- `degenerate_control_points::Array{Float64, 3}`: The degenerate control points.
- `num_basis_r::Int`: Number of radial basis functions.
- `two_poles::Bool=false`: Whether the polar spline space has two poles.
- `zero_at_poles::Bool=nothing`: Whether functions are constrained to zero at poles.

# Returns
- `E::SparseMatrixCSC{Float64,Int}`: The extraction operator.
- `tri::Matrix{Float64}`: The control triangle.
"""
function extract_scalar_polar_splines_to_tensorproduct(
    degenerate_control_points::Array{Float64, 3},
    num_basis_r::Int,
    two_poles::Bool=false,
    zero_at_poles::Bool=nothing,
)
    if size(degenerate_control_points, 3) != 2
        throw(
            ArgumentError(
                "The control points need to be two-dimensional, got $(size(degenerate_control_points, 3)).",
            ),
        )
    end
    if ~all(
        isapprox.(
            degenerate_control_points[:, 1, :],
            degenerate_control_points[1:1, 1, :],
            atol=1e-12,
        ),
    )
        throw(ArgumentError("All points in the first set should be identical."))
    else
        origin = degenerate_control_points[1, 1, :]
    end

    # number of basis functions in poloidal direction
    num_basis_p = size(degenerate_control_points, 1)

    # build triangle circumscribing the input degenerate control points
    tri = _get_circumscribing_triangle(
        vcat(degenerate_control_points[:, 1, :], degenerate_control_points[:, 2, :]), origin
    )

    if zero_at_poles
        # eliminate all functions that are non-zero at the poles
        num_basis = num_basis_p * (num_basis_r - 2 - two_poles)

        # allocate row, col, val vectors for extraction matrix
        E2_r = zeros(Int, num_basis)
        E2_c = zeros(Int, num_basis)
        E2_v = zeros(Float64, num_basis)
        ind = 0

        # generate data for extraction
        k = 0:(num_basis_p - 1)
        l = 1:(num_basis_r - 2 - two_poles)
        for idx in Iterators.product(k, l)
            r = idx[1] + (idx[2] - 1) * num_basis_p
            c = idx[1] + idx[2] * num_basis_p
            E2_r[ind + 1] = r + 1
            E2_c[ind + 1] = c + 1
            E2_v[ind + 1] = 1.0
            ind += 1
        end

        # assembly
        E = SparseArrays.sparse(
            E2_r, E2_c, E2_v, num_basis, num_basis_p * (num_basis_r - 1)
        )

    else
        # two rows of functions at each pole will be linearly combined into three
        num_basis = num_basis_p * num_basis_r - (two_poles + 1) * (2 * num_basis_p - 3)

        # compute barycentric coordinates and extraction sub-matrices for the input points
        E0_1, E0_2 = _get_scalar_polar_extraction_submatrix(
            degenerate_control_points, tri, origin, two_poles
        )

        # assemble full polar extraction matrix
        E = SparseArrays.blockdiag(
            E0_1, SparseArrays.spdiagm(ones(max(0, num_basis - 6))), E0_2
        )
        SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, E)
    end

    return E, tri
end

"""
    _get_scalar_polar_extraction_submatrix(
        degenerate_control_points::Array{Float64, 3},
        tri::Matrix{Float64},
        origin::Vector{Float64}=[0.0, 0.0],
        two_poles::Bool=false
    )

Compute the extraction sub-matrices for the scalar polar spline space. These are used to
build the scalar polar spline extraction operator, as well as the vector polar spline
extraction operator.

# Arguments
- `degenerate_control_points::Array{Float64, 3}`: The degenerate control points.
- `tri::Matrix{Float64}`: The control triangle.
- `origin::Vector{Float64}=[0.0, 0.0]`: The origin.
- `two_poles::Bool=false`: Whether the polar spline space has two poles.

# Returns
- `E0_1::SparseMatrixCSC{Float64,Int}`: The extraction sub-matrix for the first pole.
- `E0_2::SparseMatrixCSC{Float64,Int}`: The extraction sub-matrix for the second pole.
"""
function _get_scalar_polar_extraction_submatrix(
    degenerate_control_points::Array{Float64, 3},
    tri::Matrix{Float64},
    origin::Vector{Float64}=[0.0, 0.0],
    two_poles::Bool=false,
)
    # compute barycentric coordinates w.r.t. the control triangle
    baryc_1 = permutedims(
        _get_barycentric_coordinates(degenerate_control_points[:, 1, :], tri, origin)
    )
    baryc_2 = permutedims(
        _get_barycentric_coordinates(degenerate_control_points[:, 2, :], tri, origin)
    )

    # create extraction matrix for first pole
    E0_1 = SparseArrays.SparseArrays.sparse([baryc_1 baryc_2])
    if two_poles
        # create symmetric extraction matrix for second pole
        E0_2 = reverse(E0_1; dims=2)
    else
        E0_2 = SparseArrays.spdiagm(ones(3))
    end
    return E0_1, E0_2
end

"""
    extract_vector_polar_splines_to_tensorproduct(
        degenerate_control_points::Array{Float64, 3},
        num_basis_r::Int;
        two_poles::Bool=false
    )

Build the extraction operator to extract the vector polar spline basis functions w.r.t.
the tensor product basis functions.

# Arguments
- `degenerate_control_points::Array{Float64, 3}`: The degenerate control points.
- `num_basis_r::Int`: Number of radial basis functions.
- `two_poles::Bool=false`: Whether the polar spline space has two poles.

# Returns
- `E::Tuple{SparseMatrixCSC{Float64,Int},SparseMatrixCSC{Float64,Int}}`: The extraction operators.
The first element corresponds to the horizontal edges and the second element to the vertical edges.
- `tri::Matrix{Float64}`: The control triangle.
"""
function extract_vector_polar_splines_to_tensorproduct(
    degenerate_control_points::Array{Float64, 3},
    num_basis_r::Int,
    two_poles::Bool=false,
)
    if size(degenerate_control_points, 3) != 2
        throw(
            ArgumentError(
                "The control points need to be two-dimensional, got $(size(degenerate_control_points, 3)).",
            ),
        )
    end
    if ~all(
        isapprox.(
            degenerate_control_points[:, 1, :],
            degenerate_control_points[1:1, 1, :],
            atol=1e-12,
        ),
    )
        throw(ArgumentError("All points in the first set should be identical."))
    else
        origin = degenerate_control_points[1, 1, :]
    end

    # number of basis functions in poloidal direction
    num_basis_p = size(degenerate_control_points, 1)

    # build triangle circumscribing the input degenerate control points
    tri = _get_circumscribing_triangle(
        vcat(degenerate_control_points[:, 1, :], degenerate_control_points[:, 2, :]), origin
    )

    # number of vector basis functions
    num_basis =
        num_basis_p * num_basis_r + num_basis_p * (num_basis_r - 1) -
        (two_poles + 1) * (3 * num_basis_p - 2)

    # allocate row, col, val vectors for extraction matrix for vertical and horizontal edges
    E1_v_r = zeros(Int, num_basis * 4)
    E1_v_c = zeros(Int, num_basis * 4)
    E1_v_v = zeros(Float64, num_basis * 4)
    E1_h_r = zeros(Int, num_basis * 4)
    E1_h_c = zeros(Int, num_basis * 4)
    E1_h_v = zeros(Float64, num_basis * 4)
    ind_v = 0
    ind_h = 0

    # compute barycentric coordinates and extraction sub-matrices for the input points
    E0_1, E0_2 = _get_scalar_polar_extraction_submatrix(
        degenerate_control_points, tri, origin, two_poles
    )

    # assemble the extraction near the bottom pole
    baryc = hcat(
        reshape(E0_1[:, 1:num_basis_p]', num_basis_p, 1, 3),
        reshape(E0_1[:, num_basis_p .+ (1:num_basis_p)]', num_basis_p, 1, 3),
    )
    for k in 0:(num_basis_p - 1)
        l = 0
        kp = mod(k + 1, num_basis_p)
        r = [0, 1]

        c = k
        v = baryc[k + 1, l + 2, 2:3] .- baryc[k + 1, l + 1, 2:3]
        E1_v_r[ind_v .+ (1:length(r))] = r .+ 1
        E1_v_c[ind_v .+ (1:length(r))] .= c + 1
        E1_v_v[ind_v .+ (1:length(r))] = v
        ind_v += length(r)

        c = k + num_basis_p
        v = baryc[kp + 1, l + 2, 2:3] .- baryc[k + 1, l + 2, 2:3]
        E1_h_r[ind_h .+ (1:length(r))] = r .+ 1
        E1_h_c[ind_h .+ (1:length(r))] .= c + 1
        E1_h_v[ind_h .+ (1:length(r))] = v
        ind_h += length(r)
    end

    # assemble the extraction for intermediate vertical edges
    k = 0:(num_basis_p - 1)
    l = 2:(num_basis_r - two_poles - 1)
    for idx in Iterators.product(k, l)
        r = idx[1] + (2 * idx[2] - 4) * num_basis_p + 2
        c = idx[1] + (idx[2] - 1) * num_basis_p
        E1_v_r[ind_v + 1] = r + 1
        E1_v_c[ind_v + 1] = c + 1
        E1_v_v[ind_v + 1] = 1.0
        ind_v += 1
    end

    # assemble the extraction for intermediate horizontal edges
    k = 0:(num_basis_p - 1)
    l = 2:(num_basis_r + 1 - 2 * (two_poles + 1))
    for idx in Iterators.product(k, l)
        r = idx[1] + (2 * idx[2] - 3) * num_basis_p + 2
        c = idx[1] + idx[2] * num_basis_p
        E1_h_r[ind_h + 1] = r + 1
        E1_h_c[ind_h + 1] = c + 1
        E1_h_v[ind_h + 1] = 1.0
        ind_h += 1
    end

    # assemble the extraction near the top pole
    if two_poles
        baryc = hcat(
            reshape(E0_2[:, 1:num_basis_p]', num_basis_p, 1, 3),
            reshape(E0_2[:, num_basis_p .+ (1:num_basis_p)]', num_basis_p, 1, 3),
        )
        for k in 0:(num_basis_p - 1)
            kp = mod(k + 1, num_basis_p)
            l = num_basis_r - 2
            r = [num_1_forms - 2, num_1_forms - 1]
            c = k + l * num_basis_p

            v =
                baryc[k + 1, l - num_basis_r + 4, 2:3] .-
                baryc[k + 1, l - num_basis_r + 3, 2:3]
            E1_v_r[ind_v .+ (1:length(r))] = r .+ 1
            E1_v_c[ind_v .+ (1:length(r))] .= c + 1
            E1_v_v[ind_v .+ (1:length(r))] = v
            ind_v += length(r)

            v =
                baryc[kp + 1, l - num_basis_r + 3, 2:3] .-
                baryc[k + 1, l - num_basis_r + 3, 2:3]
            E1_h_r[ind_h .+ (1:length(r))] = r .+ 1
            E1_h_c[ind_h .+ (1:length(r))] .= c + 1
            E1_h_v[ind_h .+ (1:length(r))] = v
            ind_h += length(r)
        end
    end

    # assemble the full extraction matrix w.r.t. vertical tensor-product edges
    E1_v_r = E1_v_r[1:ind_v]
    E1_v_c = E1_v_c[1:ind_v]
    E1_v_v = E1_v_v[1:ind_v]
    E_v = SparseArrays.sparse(
        E1_v_r, E1_v_c, E1_v_v, num_basis, num_basis_p * (num_basis_r - 1)
    )
    SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, E_v)

    # assemble the full extraction matrix w.r.t. horizontal tensor-product edges
    E1_h_r = E1_h_r[1:ind_h]
    E1_h_c = E1_h_c[1:ind_h]
    E1_h_v = E1_h_v[1:ind_h]
    E_h = SparseArrays.sparse(E1_h_r, E1_h_c, E1_h_v, num_basis, num_basis_p * num_basis_r)
    SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, E_h)

    return (E_h, E_v), tri
end

############################################################################################
# Getters
############################################################################################

function get_local_basis(
    space::PolarSplineSpace{num_components},
    element_id::Int,
    xi::Points.AbstractPoints{2},
    nderivatives::Int,
    component_id::Int,
) where {num_components}
    if component_id > num_components
        throw(ArgumentError("PolarSplineSpace only has $num_components component."))
    end
    return evaluate(space.patch_spaces[component_id], element_id, xi, nderivatives)[1]
end

function get_num_elements_per_patch(space::PolarSplineSpace)
    return space.num_elements_per_patch
end

get_patch_spaces(space::PolarSplineSpace) = space.patch_spaces
function get_max_local_dim(space::PolarSplineSpace)
    return sum(get_max_local_dim.(get_patch_spaces(space)))
end

function assemble_global_extraction_matrix(
    space::PolarSplineSpace{num_components}
) where {num_components}
    return permutedims(SparseArrays.sparse_hcat(space.global_extraction_matrix...))
end

function get_degenerate_control_points(space::PolarSplineSpace)
    return space.degenerate_control_points
end

function get_degenerate_space(space::PolarSplineSpace)
    return space.degenerate_space
end

function get_element_lengths(space::PolarSplineSpace, element_id::Int)
    return get_element_lengths(get_patch_spaces(space)[1], element_id)
end
