"""
    TensorProductSpace{
        manifold_dim, num_components, num_patches, num_spaces, TP, G, GP, D
    } <: AbstractFESpace{manifold_dim, num_components, num_patches}

A structure representing  the tensor product of `num_spaces` factor spaces. The resulting
tensor product has a `manifold_dim` equal to the sum of the factor spaces' manifold
dimensions.

The interface with `TensorProducts` is done by defining the number of objects of a function
space as the number of basis functions; the ids then refer to basis ids. See
[`TensorProducts`](@ref).

# Fields
- `tensor_product::TP`: A `TensorProducts.TensorProduct` of the factor spaces.
- `geometry::G`: The underlying physical geometry.
- `parametric_geometry::GP`: The underlying parametric geometry. The function space is
    defined with respect to this.
- `dof_partition::D`: See [`get_dof_partition`](@ref).
"""
struct TensorProductSpace{
    manifold_dim, num_components, num_patches, num_spaces, TP, G, GP, D
} <: AbstractFESpace{manifold_dim, num_components, num_patches}
    tensor_product::TP
    geometry::G
    parametric_geometry::GP
    dof_partition::D

    function TensorProductSpace(
        spaces::T,
        geometry::Geometry.AbstractGeometry{manifold_dim_G, image_dim, num_patches_G},
        parametric_geometry::Geometry.AbstractGeometry{
            manifold_dim_G, manifold_dim_G, num_patches_G
        },
    ) where {
        manifold_dim_G,
        image_dim,
        num_patches_G,
        num_spaces,
        T <: NTuple{num_spaces, AbstractFESpace},
    }
        if all(get_num_components.(spaces) .== 1)
            num_components = 1
        else
            throw(
                ArgumentError(
                    LazyString(
                        "All factor spaces must have only one component, but got ",
                        get_num_components.(spaces),
                        "as the number of components for each space.",
                    ),
                ),
            )
        end

        tensor_product = TensorProduct(spaces)
        # Parameters for the Tensor-product space
        manifold_dim = sum(get_manifold_dim, spaces)
        num_patches = prod(get_num_patches, spaces)
        if manifold_dim != manifold_dim_G
            throw(
                ArgumentError(
                    LazyString(
                        "The sum of the `manifold_dim`s of each of the ",
                        "`factor_spaces` must match the `manifold_dim` of the ",
                        "`geometry`, but got ",
                        manifold_dim,
                        "and ",
                        manifold_dim_G,
                        ", resprectively.",
                    ),
                ),
            )
        end

        if num_patches != num_patches_G
            throw(
                ArgumentError(
                    LazyString(
                        "The product of the `num_patches` of each of the ",
                        "`factor_spaces` must match the `num_patches` of the ",
                        "`geometry`, but got ",
                        num_patches,
                        "and ",
                        num_patches_G,
                        ", resprectively.",
                    ),
                ),
            )
        end

        # Pre-allocate memory for degree of freedom partitioning
        dof_partition = Vector{Vector{Vector{Int}}}(undef, num_patches)
        # factor spaces
        factor_dof_partitions = map(get_dof_partition, spaces)
        factor_num_patches = map(length, factor_dof_partitions)
        lin_num_basis = TensorProducts.get_lin_ids(tensor_product)
        # Loop over all spaces and build the appropriate index subsets
        for (patch_count, patch_ids) in enumerate(CartesianIndices(factor_num_patches))
            factor_patches = ntuple(
                space -> factor_dof_partitions[space][patch_ids[space]], num_spaces
            )
            factor_partition_lengths = ntuple(
                space -> length(factor_patches[space]), num_spaces
            )
            dof_partition[patch_count] = Vector{Vector{Int}}(
                undef, prod(factor_partition_lengths)
            )
            for (partition_count, partition_ids) in
                enumerate(CartesianIndices(factor_partition_lengths))
                factor_partition_dofs = ntuple(
                    space -> factor_patches[space][partition_ids[space]], num_spaces
                )
                dof_partition[patch_count][partition_count] = Vector{Int}(
                    undef, prod(length, factor_partition_dofs)
                )
                for (dof_count, factor_basis_id) in
                    enumerate(Iterators.product(factor_partition_dofs...))
                    basis_id = lin_num_basis[factor_basis_id...]
                    dof_partition[patch_count][partition_count][dof_count] = basis_id
                end
            end
        end

        return new{
            manifold_dim,
            num_components,
            num_patches,
            num_spaces,
            typeof(tensor_product),
            typeof(geometry),
            typeof(parametric_geometry),
            typeof(dof_partition),
        }(
            tensor_product, geometry, parametric_geometry, dof_partition
        )
    end

    function TensorProductSpace(
        factor_spaces::T
    ) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
        # Create a tensor-product geometry from the factor ones.
        factor_geometries = map(get_geometry, factor_spaces)
        geometry = Geometry.TensorProductGeometry(factor_geometries)
        factor_parametric_geometries = map(get_parametric_geometry, factor_spaces)
        parametric_geometry = Geometry.TensorProductGeometry(factor_parametric_geometries)
        return TensorProductSpace(factor_spaces, geometry, parametric_geometry)
    end

    function TensorProductSpace(
        factor_spaces::T, mapping::Geometry.Mapping
    ) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
        factor_geometries = map(get_geometry, factor_spaces)
        geometry = Geometry.TensorProductGeometry(factor_geometries)
        return TensorProductSpace(
            factor_spaces, Geometry.MappedGeometry(geometry, mapping), geometry
        )
    end

    function TensorProductSpace(
        factor_spaces::T, ::Type{G}
    ) where {
        num_spaces,
        T <: NTuple{num_spaces, AbstractFESpace},
        G <: Geometry.CartesianGeometry,
    }
        factor_geometries = map(get_geometry, factor_spaces)
        geometry = Geometry.TensorProductGeometry(factor_geometries)
        cartesian_geometry = convert(G, geometry)
        return TensorProductSpace(factor_spaces, cartesian_geometry, cartesian_geometry)
    end

    function TensorProductSpace(
        factor_spaces::T, ::Type{G}, mapping::Geometry.AbstractMapping
    ) where {
        num_spaces,
        T <: NTuple{num_spaces, AbstractFESpace},
        G <: Geometry.CartesianGeometry,
    }
        factor_geometries = map(get_geometry, factor_spaces)
        geometry = Geometry.TensorProductGeometry(factor_geometries)
        cartesian_geometry = convert(G, geometry)
        return TensorProductSpace(
            factor_spaces,
            Geometry.MappedGeometry(cartesian_geometry, mapping),
            cartesian_geometry,
        )
    end
end

TensorProductSpace(spaces...) = TensorProductSpace(spaces)

TensorProducts.get_factors(space::TensorProductSpace) = get_factor_spaces(space)

"""
    get_tensor_product(space::TensorProductSpace)

Return the `TensorProducts.TensorProduct` object of the factor spaces of `space`.
"""
get_tensor_product(space::TensorProductSpace) = space.tensor_product

"""
    get_factor_spaces(space::TensorProductSpace)

Return the factor spaces of `space`.
"""
function get_factor_spaces(space::TensorProductSpace)
    return TensorProducts.get_factors(get_tensor_product(space))
end

function get_num_basis(space::TensorProductSpace)
    #=
    The `TensorProducts` module already implements a `get_num_objects` by checking the size
    of the `CartesianIndices` iterator.
    However, `TensorProducts` needs to call `get_num_objects` on the factor geometries, which
    is why there is also a `TensorProducts.get_num_objects(space::AbstractFESpace)`.
    =#
    return TensorProducts.get_num_objects(get_tensor_product(space))
end

"""
	get_cart_num_elements(space::TensorProductSpace)

See [`Geometry.get_cart_num_elements`](@ref).
"""
function get_cart_num_elements(space::TensorProductSpace)
    return Geometry.get_cart_num_elements(get_parametric_geometry(space))
end

"""
	get_lin_num_elements(space::TensorProductSpace)

See [`Geometry.get_lin_num_elements`](@ref).
"""
function get_lin_num_elements(space::TensorProductSpace)
    return Geometry.get_lin_num_elements(get_parametric_geometry(space))
end

"""
    get_num_spaces(
        ::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns the number of factor spaces in a given `TensorProductSpace`.
"""
function get_num_spaces(
    ::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
) where {manifold_dim, num_components, num_patches, num_spaces}
    return num_spaces
end

"""
    get_factor_element_ids(space::TensorProductSpace, element_id::Int)

Returns a tuple corresponding to the conversion of `element_id` in tensor-product numbering
to factor-wise numbering.
"""
function get_factor_element_ids(space::TensorProductSpace, element_id::Int)
    return Geometry.get_factor_element_ids(get_parametric_geometry(space), element_id)
end

"""
    get_cart_num_basis(space::TensorProductSpace)

Return a `CartesianIndices` iterator over the number of basis functions in `space`.
"""
function get_cart_num_basis(space::TensorProductSpace)
    return TensorProducts.get_cart_ids(get_tensor_product(space))
end

"""
    get_lin_num_basis(space::TensorProductSpace)

Return a `LinearIndices` iterator over the number of basis functions in `space`.
"""
function get_lin_num_basis(space::TensorProductSpace)
    return TensorProducts.get_lin_ids(get_tensor_product(space))
end

function TensorProducts.get_factor_ids(space::TensorProductSpace, basis_id::Int)
    return get_factor_basis_ids(space, basis_id)
end

"""
    get_factor_basis_ids(space::TensorProductSpace, basis_id::Int)

Returns a tuple corresponding to the conversion of `basis_id` in tensor-product numbering to
factor-wise numbering.
"""
function get_factor_basis_ids(space::TensorProductSpace, basis_id::Int)
    return Tuple(get_cart_num_basis(space)[basis_id])
end

"""
    get_factor_num_basis(space::TensorProductSpace)

Returns a tuple corresponding to the factor-wise number of basis functions — or
dimension.
"""
function get_factor_num_basis(space::TensorProductSpace)
    return TensorProducts.get_factor_num_objects(get_tensor_product(space))
end

"""
    get_factor_num_basis(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
        element_id::Int,
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the factor-wise number of basis functions supported on
`element_id`.
"""
function get_factor_num_basis(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    factor_element_ids, _ = get_factor_element_ids(space, element_id)

    return map(get_num_basis, get_factor_spaces(space), factor_element_ids)
end

function get_num_basis(space::TensorProductSpace, element_id::Int)
    return prod(get_factor_num_basis(space, element_id))
end

"""
    get_factor_manifold_dims(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the factor-wise manifold dimension.
"""
function get_factor_manifold_dims(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
) where {manifold_dim, num_components, num_patches, num_spaces}
    return TensorProducts.mapfactors(get_manifold_dim, get_tensor_product(space))
end

"""
    get_factor_basis_indices( space::TensorProductSpace, element_id::Int)

Returns a tuple corresponding to the factor-wise basis indices supported on `element_id`.

See also [`get_basis_indices`](@ref).
"""
function get_factor_basis_indices(space::TensorProductSpace, element_id::Int)
    factor_element_ids, _ = get_factor_element_ids(space, element_id)

    return map(get_basis_indices, get_factor_spaces(space), factor_element_ids)
end

"""
    get_factor_supports(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
        basis_id::Int,
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the factor-wise support of the basis function
identified by `basis_id`.

See also [`get_support`](@ref).
"""
function get_factor_supports(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    basis_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    return TensorProducts.mapfactors(get_support, get_tensor_product(space), basis_id)
end

"""
    get_factor_extraction(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
        element_id::Int,
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the factor-wise extraction at `element_id`.

See also [`get_extraction`](@ref).
"""
function get_factor_extractions(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    factor_element_ids, _ = get_factor_element_ids(space, element_id)

    return map(get_extraction, get_factor_spaces(space), factor_element_ids)
end

"""
    get_factor_polynomial_degrees(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the factor-wise polynomial degree. Note that
`get_polynomial_degree` is not necessarily defined for every factor space.

See also [`get_polynomial_degree`](@ref).
"""
function get_factor_polynomial_degrees(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
) where {manifold_dim, num_components, num_patches, num_spaces}
    return TensorProducts.mapfactors(get_polynomial_degree, get_tensor_product(space))
end

"""
    get_factor_local_basis(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
        nderivatives::Int=0,
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the factor-wise local basis evaluation at `element_id`
and points `xi`.

See also [`get_local_basis`](@ref).
"""
function get_factor_local_basis(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int=0,
) where {manifold_dim, num_components, num_patches, num_spaces}
    factor_spaces = get_factor_spaces(space)
    factor_element_id, _ = get_factor_element_ids(space, element_id)
    factor_xi = get_factor_evaluation_points(space, xi)
    factor_local_basis = ntuple(
        space -> get_local_basis(
            factor_spaces[space],
            factor_element_id[space],
            factor_xi[space],
            nderivatives,
        ),
        num_spaces,
    )

    return factor_local_basis
end

"""
    get_factor_evaluations(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
        nderivatives::Int=0,
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Get evaluations of all factor spaces at `element_id` and points `xi`.
"""
function get_factor_evaluations(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int=0,
) where {manifold_dim, num_components, num_patches, num_spaces}
    factor_spaces = get_factor_spaces(space)
    factor_element_id, _ = get_factor_element_ids(space, element_id)
    factor_xi = get_factor_evaluation_points(space, xi)
    tup_ders = ntuple(i -> nderivatives, num_spaces)
    factor_eval_and_inds = map(
        evaluate, factor_spaces, factor_element_id, factor_xi, tup_ders
    )
    factor_eval = map(getindex, factor_eval_and_inds, ntuple(i -> 1, num_spaces))

    return factor_eval
end

"""
    get_factor_evaluation_points(
        space::TensorProductSpace{manifold_dim}, xi::Points.AbstractPoints{manifold_dim}
    ) where {manifold_dim}

See [`Geometry.get_factor_evaluation_points`](@ref).
"""
function get_factor_evaluation_points(
    space::TensorProductSpace{manifold_dim}, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    return Geometry.get_factor_evaluation_points(get_parametric_geometry(space), xi)
end

"""
    get_factor_manifold_indices(space::TensorProductSpace)

See [`Geometry.get_factor_manifold_indices`](@ref).
"""
function get_factor_manifold_indices(space::TensorProductSpace)
    return Geometry.get_factor_manifold_indices(get_parametric_geometry(space))
end

function get_factor_element_vertices(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    return Geometry.get_factor_element_vertices(get_parametric_geometry(space), element_id)
end

function get_factor_element_lengths(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    return Geometry.get_factor_element_lengths(get_parametric_geometry(space), element_id)
end

function get_basis_indices(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    factor_basis_indices = get_factor_basis_indices(space, element_id)
    product = Iterators.product(factor_basis_indices...)
    lin_num_basis = get_lin_num_basis(space)
    basis_indices = Vector{Int}(undef, length(product))
    for (i, basis) in enumerate(product)
        basis_indices[i] = lin_num_basis[basis...]
    end

    return basis_indices
end

function get_support(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    basis_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    factor_supports = get_factor_supports(space, basis_id)
    product = Iterators.product(factor_supports...)
    lin_num_elements = get_lin_num_elements(space)
    support = Iterators.flatten(Iterators.map(e -> lin_num_elements[e...], product))

    return support
end

function get_max_local_dim(space::TensorProductSpace)
    return prod(get_max_local_dim, get_factor_spaces(space))
end

function get_extraction_coefficients(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    component_id::Int=1,
) where {manifold_dim, num_components, num_patches, num_spaces}
    # The permutations of the factor spaces should be combined if we allow for more
    # than one component.
    extraction_per_space = get_factor_extractions(space, element_id)
    if num_spaces == 1
        extraction_coeffs = extraction_per_space[1][1]
    elseif all([
        typeof(eps[1]) <: LinearAlgebra.UniformScaling for eps in extraction_per_space
    ])
        extraction_coeffs = LinearAlgebra.I
    else
        extraction_coeffs = kron(
            (extraction_per_space[space_id][1] for space_id in num_spaces:-1:1)...
        )
    end

    return extraction_coeffs
end

function get_extraction(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    component_id::Int=1,
) where {manifold_dim, num_components, num_patches, num_spaces}
    # We cannot call size on the extraction coefficients intead of get_basis_permutation
    # because some extraction coefficients are LinearAlgebra.UniformScaling (the identity),
    # for which size is not defined.
    return (
        get_extraction_coefficients(space, element_id, component_id),
        get_basis_permutation(space, element_id, component_id),
    )
end

function get_basis_permutation(
    space::TensorProductSpace, element_id::Int, component_id::Int=1
)
    return 1:get_num_basis(space, element_id)
end

function get_local_basis(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int=0,
    component_id::Int=1,
) where {manifold_dim, num_components, num_patches, num_spaces}
    factor_local_basis = get_factor_local_basis(space, element_id, xi, nderivatives)
    factor_sizes = ntuple(space -> size(factor_local_basis[space][1][1][1]), num_spaces)
    num_points = Points.get_num_points(xi)
    num_basis = get_num_basis(space, element_id)
    local_basis = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
    for der_order in 0:nderivatives
        num_der_ids = GeneralHelpers.num_der_indices(manifold_dim, der_order)
        # We assume that there is only one component.
        local_basis[der_order + 1] = [
            [Matrix{Float64}(undef, (num_points, num_basis))] for _ in 1:num_der_ids
        ]
    end

    der_keys = integer_sums(0, nderivatives, Val(manifold_dim))
    factor_manifold_indices = get_factor_manifold_indices(space)
    der_keys = integer_sums(0, nderivatives, Val(manifold_dim))
    factor_eval = ntuple(space -> Matrix{Float64}(undef, factor_sizes[space]), num_spaces)
    space_der_order = zeros(Int, num_spaces)
    space_der_id = zeros(Int, num_spaces)
    for key in der_keys
        der_order = sum(key)
        der_id = get_derivative_idx(key)
        for space in 1:num_spaces
            space_der_order, space_der_id = _get_key_info(
                key, factor_manifold_indices, space
            )
            space_eval = factor_local_basis[space][space_der_order + 1][space_der_id][1]
            factor_eval[space] .= space_eval
        end

        if manifold_dim == 1
            local_basis[der_order + 1][der_id][1] = factor_eval[1]
        else
            local_basis[der_order + 1][der_id][1] = kron(
                (factor_eval[space] for space in num_spaces:-1:1)...
            )
        end
    end

    return local_basis
end

function get_factor_derivative_key(
    key::NTuple{manifold_dim, Int}, factor_indices, factor_id::Int
) where {manifold_dim}
    key_size = length(factor_indices[factor_id])
    factor_key = ntuple(i -> key[factor_indices[factor_id][i]], key_size)

    return factor_key
end

function evaluate(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    xi::Points.TensorProductPoints{manifold_dim},
    nderivatives::Int=0,
) where {manifold_dim, num_components, num_patches, num_spaces}
    basis_indices = get_basis_indices(space, element_id)
    num_basis = length(basis_indices)
    factor_eval = get_factor_evaluations(space, element_id, xi, nderivatives)
    num_points = Points.get_num_points(xi)
    eval = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
    for der_order in 0:nderivatives
        num_der_ids = GeneralHelpers.num_der_indices(manifold_dim, der_order)
        # We assume that there is only one component.
        eval[der_order + 1] = [
            [Matrix{Float64}(undef, (num_points, num_basis))] for _ in 1:num_der_ids
        ]
    end

    factor_manifold_indices = get_factor_manifold_indices(space)
    der_keys = integer_sums(0, nderivatives, Val(manifold_dim))
    space_der_order = zeros(Int, num_spaces)
    space_der_id = zeros(Int, num_spaces)
    for key in der_keys
        der_order = sum(key)
        der_id = get_derivative_idx(key)
        _update_key_info!(space_der_order, space_der_id, key, factor_manifold_indices)
        if num_spaces == 1
            eval[der_order + 1][der_id][1] .= factor_eval[1][space_der_order[1] + 1][space_der_id[1]][1]
        else
            eval[der_order + 1][der_id][1] .= kron(
                (
                    factor_eval[space][space_der_order[space] + 1][space_der_id[space]][1]
                    for space in num_spaces:-1:1
                )...,
            )
        end
    end

    return eval, basis_indices
end

function evaluate(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int=0,
) where {manifold_dim, num_components, num_patches, num_spaces}
    basis_indices = get_basis_indices(space, element_id)
    num_basis = length(basis_indices)
    factor_eval = get_factor_evaluations(space, element_id, xi, nderivatives)
    num_points = Points.get_num_points(xi)
    eval = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
    for der_order in 0:nderivatives
        num_der_ids = GeneralHelpers.num_der_indices(manifold_dim, der_order)
        # We assume that there is only one component.
        eval[der_order + 1] = [
            [Matrix{Float64}(undef, (num_points, num_basis))] for _ in 1:num_der_ids
        ]
    end

    factor_manifold_indices = get_factor_manifold_indices(space)
    der_keys = integer_sums(0, nderivatives, Val(manifold_dim))
    space_der_order = zeros(Int, num_spaces)
    space_der_id = zeros(Int, num_spaces)
    for key in der_keys
        der_order = sum(key)
        der_id = get_derivative_idx(key)
        _update_key_info!(space_der_order, space_der_id, key, factor_manifold_indices)
        if num_spaces == 1
            eval[der_order + 1][der_id][1] .= factor_eval[1][space_der_order[1] + 1][space_der_id[1]][1]
        else
            for point in axes(eval[der_order + 1][der_id][1], 1)
                eval[der_order + 1][der_id][1][point, :] .= kron(
                    (
                        factor_eval[space][space_der_order[space] + 1][space_der_id[space]][1][
                            point, :,
                        ] for space in num_spaces:-1:1
                    )...,
                )
            end
        end
    end

    return eval, basis_indices
end

function _update_key_info!(space_der_order, space_der_id, key, factor_manifold_indices)
    for space in eachindex(space_der_order, space_der_id)
        der_order, der_id = _get_key_info(key, factor_manifold_indices, space)
        space_der_order[space] = der_order
        space_der_id[space] = der_id
    end

    return space_der_order, space_der_id
end

function _get_key_info(key, factor_manifold_indices, space)
    factor_key = get_factor_derivative_key(key, factor_manifold_indices, space)
    space_der_order = sum(factor_key)
    space_der_id = get_derivative_idx(factor_key)

    return space_der_order, space_der_id
end

# Methods for tensor product B-spline spaces
include("BSplines.jl")
