"""
    TensorProductSpace{
        manifold_dim, num_components, num_patches, num_spaces, T, G, GP, CIB, LIB, D
    } <: AbstractFESpace{manifold_dim, num_components, num_patches}

A structure representing a `TensorProductSpace`, defined by the tensor product of
`num_spaces` constituent spaces. The resulting tensor product has a `manifold_dim` equal to
the sum of the constituent spaces' manifold dimensions.

# Fields
- `constituent_spaces::T`: A tuple of constituent finite element spaces to be tensored.
- `geometry::G`: The underlying physical geometry.
- `parametric_geometry::GP`: The underlying parametric geometry. The function space is
    defined with respect to this.
- `cart_num_basis::CIB`: To convert from tensor-product indexing to constituent-wise
    indexing for basis functions.
- `lin_num_basis::LIB`: To convert from constituent-wise indexing to tensor-product indexing
    for basis functions.
- `dof_partition::D`: See [`get_dof_partition`](@ref).
"""
struct TensorProductSpace{
    manifold_dim, num_components, num_patches, num_spaces, T, G, GP, CIB, LIB, D
} <: AbstractFESpace{manifold_dim, num_components, num_patches}
    constituent_spaces::T
    geometry::G
    parametric_geometry::GP
    cart_num_basis::CIB
    lin_num_basis::LIB
    dof_partition::D

    function TensorProductSpace(
        constituent_spaces::T,
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
        if all(get_num_components.(constituent_spaces) .== 1)
            num_components = 1
        else
            throw(
                ArgumentError(
                    LazyString(
                        "All input spaces must have only one component, but got ",
                        get_num_components.(constituent_spaces),
                        "as the number of components for each space.",
                    ),
                ),
            )
        end

        # Parameters for the Tensor-product space
        manifold_dim = sum(get_manifold_dim, constituent_spaces)
        num_patches = prod(get_num_patches, constituent_spaces)
        if manifold_dim != manifold_dim_G
            throw(
                ArgumentError(
                    LazyString(
                        "The sum of the `manifold_dim`s of each of the ",
                        "`constituent_spaces` must match the `manifold_dim` of the ",
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
                        "`constituent_spaces` must match the `num_patches` of the ",
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
        # Constituent spaces
        const_dof_partitions = ntuple(
            space -> get_dof_partition(constituent_spaces[space]), num_spaces
        )
        const_num_patches = ntuple(space -> length(const_dof_partitions[space]), num_spaces)
        const_num_basis = ntuple(
            space -> get_num_basis(constituent_spaces[space]), num_spaces
        )
        cart_num_basis = CartesianIndices(const_num_basis)
        lin_num_basis = LinearIndices(const_num_basis)
        # Loop over all spaces and build the appropriate index subsets
        for (patch_count, patch_ids) in enumerate(CartesianIndices(const_num_patches))
            const_patches = ntuple(
                space -> const_dof_partitions[space][patch_ids[space]], num_spaces
            )
            const_partition_lengths = ntuple(
                space -> length(const_patches[space]), num_spaces
            )
            dof_partition[patch_count] = Vector{Vector{Int}}(
                undef, prod(const_partition_lengths)
            )
            for (partition_count, partition_ids) in
                enumerate(CartesianIndices(const_partition_lengths))
                const_partition_dofs = ntuple(
                    space -> const_patches[space][partition_ids[space]], num_spaces
                )
                dof_partition[patch_count][partition_count] = Vector{Int}(
                    undef, prod(length, const_partition_dofs)
                )
                for (dof_count, const_basis_id) in
                    enumerate(Iterators.product(const_partition_dofs...))
                    basis_id = lin_num_basis[const_basis_id...]
                    dof_partition[patch_count][partition_count][dof_count] = basis_id
                end
            end
        end

        return new{
            manifold_dim,
            num_components,
            num_patches,
            num_spaces,
            T,
            typeof(geometry),
            typeof(parametric_geometry),
            typeof(cart_num_basis),
            typeof(lin_num_basis),
            typeof(dof_partition),
        }(
            constituent_spaces,
            geometry,
            parametric_geometry,
            cart_num_basis,
            lin_num_basis,
            dof_partition,
        )
    end

    function TensorProductSpace(
        constituent_spaces::T
    ) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
        # Create a tensor-product geometry from the constituent ones.
        constituent_geometries = map(get_geometry, constituent_spaces)
        geometry = Geometry.TensorProductGeometry(constituent_geometries)
        constituent_parametric_geometries = map(get_parametric_geometry, constituent_spaces)
        parametric_geometry = Geometry.TensorProductGeometry(
            constituent_parametric_geometries
        )
        return TensorProductSpace(constituent_spaces, geometry, parametric_geometry)
    end

    function TensorProductSpace(
        constituent_spaces::T, geometry::Geometry.CartesianGeometry
    ) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
        # Check the compatibility of the underlying geometries of the constituent spaces
        # and the given CartesianGeometry (which acts as both parametric and physical
        # geometry).
        if check_compatible(constituent_spaces, geometry)
            return TensorProductSpace(constituent_spaces, geometry, geometry)
        else
            error("The geometry and spaces are unexpectedly incompatible. This is a bug.")
        end
    end

    function TensorProductSpace(
        constituent_spaces::T,
        parametric_geometry::Geometry.CartesianGeometry,
        map::Geometry.Mapping,
    ) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
        if check_compatible(constituent_spaces, parametric_geometry)
            return TensorProductSpace(
                constituent_spaces,
                Geometry.MappedGeometry(parametric_geometry, map),
                parametric_geometry,
            )
        else
            error("The geometry and spaces are unexpectedly incompatible. This is a bug.")
        end
    end

    function TensorProductSpace(
        constituent_spaces::T, geometry::Geometry.TensorProductGeometry
    ) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
        # Create a tensor-product geometry from the constituent ones, since the given
        # physical TPGeometry may consist of mapped or other geometries.
        constituent_parametric_geometries = map(get_parametric_geometry, constituent_spaces)
        parametric_geometry = Geometry.TensorProductGeometry(
            constituent_parametric_geometries
        )
        return TensorProductSpace(constituent_spaces, geometry, parametric_geometry)
    end
end

merge_two_tuples(tup::NTuple{N, T}, tup2::NTuple{N2, T}) where {N, N2, T} =
    (tup..., tup2...)
function merge_tuples(tup::NTuple{N, T}, tups...) where {N, T}
    if length(tups) == 1
        return merge_two_tuples(tup, first(tups))
    end
    return merge_tuples(merge_two_tuples(tup, first(tups)), Base.tail(tups)...)
end

function check_compatible(
    constituent_spaces::T, geometry::Geometry.CartesianGeometry
) where {num_spaces, T <: NTuple{num_spaces, AbstractFESpace}}
    # Check that the new manifold dim and number of patches matches.
    new_manifold_dim = sum(get_manifold_dim, constituent_spaces)
    if new_manifold_dim != Geometry.get_manifold_dim(geometry)
        throw(
            ArgumentError(
                LazyString(
                    "The given CartesianGeometry is not compatible with the given",
                    " spaces. The sum of the manifold dimensions of the spaces is ",
                    new_manifold_dim,
                    " while the given CartesianGeometry has a manifold dimension of ",
                    Geometry.get_manifold_dim(geometry),
                    ".",
                ),
            ),
        )
    end
    new_num_patches = prod(get_num_patches, constituent_spaces)
    if new_num_patches != Geometry.get_num_patches(geometry)
        throw(
            ArgumentError(
                LazyString(
                    "The given CartesianGeometry is not compatible with the given",
                    " spaces. The product of the number of patches of the spaces is ",
                    new_num_patches,
                    " while the given CartesianGeometry has ",
                    Geometry.get_num_patches(geometry),
                    " patches.",
                ),
            ),
        )
    end

    constituent_geometries = map(get_geometry, constituent_spaces)
    # Check that all given spaces are build using CartesianGeometries, otherwise we
    # cannot check the compatibility.
    is_cartesian = map(i -> i isa Geometry.CartesianGeometry, constituent_geometries)
    if !all(is_cartesian)
        throw(
            ArgumentError(
                LazyString(
                    "The given CartesianGeometry is not compatible with the given",
                    " spaces. The space at index ",
                    findfirst(isequal(false), is_cartesian),
                    " does not have an underlying CartesianGeometry.",
                ),
            ),
        )
    end

    # Check that the total number of elements match.
    constituent_num_elements = map(Geometry.get_num_elements, constituent_geometries)
    tp_num_elements = prod(constituent_num_elements)
    if tp_num_elements != Geometry.get_num_elements(geometry)
        throw(
            ArgumentError(
                LazyString(
                    "The given CartesianGeometry is not compatible with the given",
                    " spaces. The product of the number of elements of the spaces is ",
                    tp_num_elements,
                    " while the given CartesianGeometry has ",
                    Geometry.get_num_elements(geometry),
                    " elements.",
                ),
            ),
        )
    end

    # Check that the structure of the elements matches.
    cart_num_elements = CartesianIndices(constituent_num_elements)
    constituent_num_patches = map(Geometry.get_num_patches, constituent_geometries)
    cart_num_patches = CartesianIndices(constituent_num_patches)
    for ((patch_id, local_patch_ids), (global_element_id, local_element_ids)) in
        zip(enumerate(cart_num_patches), enumerate(cart_num_elements))
        patch_constituent_element_ids, cart_patch_id = Geometry.get_constituent_element_id(
            geometry, global_element_id
        )

        if !(patch_id == cart_patch_id)
            throw(
                ArgumentError(
                    LazyString(
                        "The given CartesianGeometry is not compatible with the given",
                        " spaces. On global element ",
                        global_element_id,
                        ", the tensor product spaces would be on patch ",
                        patch_id,
                        ", while the given CartesianGeometry is on patch ",
                        cart_patch_id,
                        ".",
                    ),
                ),
            )
        end

        elements_and_patches = map(
            Geometry.get_constituent_element_id,
            constituent_geometries,
            Tuple(local_element_ids),
        )
        elements_per_patch_constituent_ci = map(
            getindex, elements_and_patches, ntuple(i -> 1, num_spaces)
        )
        elements_per_patch_constituent_tuptup = map(
            Tuple, elements_per_patch_constituent_ci
        )
        elements_per_patch_constituent_tup = merge_tuples(
            elements_per_patch_constituent_tuptup...
        )
        if !all(elements_per_patch_constituent_tup .== Tuple(patch_constituent_element_ids))
            throw(
                ArgumentError(
                    LazyString(
                        "The given CartesianGeometry is not compatible with the given",
                        " spaces. On global element ",
                        global_element_id,
                        ", the tensor product spaces would be on local elements ",
                        elements_per_patch_constituent_tup,
                        ", while the given CartesianGeometry is on ",
                        Tuple(patch_constituent_element_ids),
                        ".",
                    ),
                ),
            )
        end
    end

    return true
end

# Getters
get_cart_num_basis(space::TensorProductSpace) = space.cart_num_basis
get_lin_num_basis(space::TensorProductSpace) = space.lin_num_basis
get_constituent_spaces(space::TensorProductSpace) = space.constituent_spaces
get_num_basis(space::TensorProductSpace) = prod(get_constituent_num_basis(space))

"""
    get_num_spaces(
        ::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns the number of constituent spaces in a given `TensorProductSpace`.
"""
function get_num_spaces(
    ::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
) where {manifold_dim, num_components, num_patches, num_spaces}
    return num_spaces
end

"""
    get_constituent_space(space::TensorProductSpace, space_id::Int)

Returns the constituent space numbered `space_id` of the tensor product `space`.
"""
function get_constituent_space(space::TensorProductSpace, space_id::Int)
    return get_constituent_spaces(space)[space_id]
end

"""
    get_constituent_basis_id(space::TensorProductSpace, basis_id::Int)

Returns a tuple corresponding to the conversion of `basis_id` in tensor-product numbering to
constituent-wise numbering.
"""
function get_constituent_basis_id(space::TensorProductSpace, basis_id::Int)
    return Tuple(get_cart_num_basis(space)[basis_id])
end

"""
    get_constituent_num_basis(space::TensorProductSpace)

Returns a tuple corresponding to the constituent-wise number of basis functions — or
dimension.
"""
function get_constituent_num_basis(space::TensorProductSpace)
    return Tuple(maximum(get_cart_num_basis(space)))
end

"""
    get_constituent_num_basis(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
        element_id::Int,
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the constituent-wise number of basis functions supported on
`element_id`.
"""
function get_constituent_num_basis(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)
    const_element_id, patch_id = Geometry.get_constituent_element_id(
        get_parametric_geometry(space), element_id
    )
    const_num_basis = map(get_num_basis, const_spaces, Tuple(const_element_id))

    return const_num_basis
end

function get_num_basis(space::TensorProductSpace, element_id::Int)
    return prod(get_constituent_num_basis(space, element_id))
end

"""
    get_constituent_manifold_dim(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the constituent-wise manifold dimension.
"""
function get_constituent_manifold_dim(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)

    return ntuple(space -> get_manifold_dim(const_spaces[space]), num_spaces)
end

"""
    get_constituent_basis_indices(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
        element_id::Int,
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the constituent-wise basis indices supported on
`element_id`.

See also [`get_basis_indices`](@ref).
"""
function get_constituent_basis_indices(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)
    const_element_id, patch_id = Geometry.get_constituent_element_id(
        get_parametric_geometry(space), element_id
    )
    const_basis_indices = map(get_basis_indices, const_spaces, Tuple(const_element_id))

    return const_basis_indices
end

"""
    get_constituent_support(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
        basis_id::Int,
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the constituent-wise support of the basis function
identified by `basis_id`.

See also [`get_support`](@ref).
"""
function get_constituent_support(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    basis_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_basis_id = get_constituent_basis_id(space, basis_id)
    const_spaces = get_constituent_spaces(space)
    const_support = ntuple(
        space -> get_support(const_spaces[space], const_basis_id[space]), num_spaces
    )

    return const_support
end

"""
    get_constituent_extraction(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
        element_id::Int,
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the constituent-wise extraction at `element_id`.

See also [`get_extraction`](@ref).
"""
function get_constituent_extraction(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_element_id, patch_id = Geometry.get_constituent_element_id(
        get_parametric_geometry(space), element_id
    )
    const_spaces = get_constituent_spaces(space)
    const_extraction = map(get_extraction, const_spaces, Tuple(const_element_id))

    return const_extraction
end

"""
    get_constituent_polynomial_degree(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the constituent-wise polynomial degree. Note that
`get_polynomial_degree` is not necessarily defined for every constituent space.

See also [`get_polynomial_degree`](@ref).
"""
function get_constituent_polynomial_degree(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)
    const_polynomial_degree = ntuple(
        space -> get_polynomial_degree(const_spaces[space]), num_spaces
    )

    return const_polynomial_degree
end

"""
    get_constituent_local_basis(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
        nderivatives::Int=0,
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the constituent-wise local basis evaluation at `element_id`
and points `xi`.

See also [`get_local_basis`](@ref).
"""
function get_constituent_local_basis(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int=0,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)
    const_element_id, patch_id = Geometry.get_constituent_element_id(
        get_parametric_geometry(space), element_id
    )
    const_xi = get_constituent_evaluation_points(space, xi)
    const_local_basis = map(
        get_local_basis, const_spaces, Tuple(const_element_id), const_xi, Ref(nderivatives)
    )

    return const_local_basis
end

"""
    get_constituent_evaluations(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
        nderivatives::Int=0,
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Get evaluations of all constituent spaces at `element_id` and points `xi`.
"""
function get_constituent_evaluations(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int=0,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)
    const_element_id, patch_id = Geometry.get_constituent_element_id(
        get_parametric_geometry(space), element_id
    )
    const_xi = get_constituent_evaluation_points(space, xi)

    tup_ders = ntuple(i -> nderivatives, num_spaces)
    const_eval_and_inds = map(
        evaluate, const_spaces, Tuple(const_element_id), const_xi, tup_ders
    )

    const_eval = map(getindex, const_eval_and_inds, ntuple(i -> 1, num_spaces))

    return const_eval
end

"""
    get_constituent_evaluation_points(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple corresponding to the constituent-wise splitting of `xi` according to each
constituent space's manifold dimension.
"""
function get_constituent_evaluation_points(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_manifold_indices = get_constituent_manifold_indices(space)
    const_points = Points.get_constituent_points(xi)
    const_xi = ntuple(
        space -> Points.CartesianPoints(const_points[const_manifold_indices[space]]),
        num_spaces,
    )

    return const_xi
end

"""
    get_constituent_manifold_indices(
        space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
    ) where {manifold_dim, num_components, num_patches, num_spaces}

Returns a tuple of ranges corresponding to the constituent-wise splitting `manifold_dim`,
use to iterate over each manifold dimension of each constituent space.
"""
function get_constituent_manifold_indices(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces}
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_manifold_dim = get_constituent_manifold_dim(space)
    cum_const_manifold_dim = (0, cumsum(const_manifold_dim)...)
    const_manifold_indices = ntuple(
        space -> (cum_const_manifold_dim[space] + 1):cum_const_manifold_dim[space + 1],
        num_spaces,
    )

    return const_manifold_indices
end

function get_constituent_element_vertices(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)
    const_element_id, patch_id = Geometry.get_constituent_element_id(
        get_parametric_geometry(space), element_id
    )
    const_element_vertices = map(
        get_element_vertices, const_spaces, Tuple(const_element_id)
    )

    return const_element_vertices
end

function get_constituent_element_lengths(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_spaces = get_constituent_spaces(space)
    const_element_id, patch_id = Geometry.get_constituent_element_id(
        get_parametric_geometry(space), element_id
    )
    const_element_lengths = map(get_element_lengths, const_spaces, Tuple(const_element_id))

    return const_element_lengths
end

function get_basis_indices(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_basis_indices = get_constituent_basis_indices(space, element_id)
    lin_num_basis = get_lin_num_basis(space)
    basis_indices = Vector{Int}(undef, prod(length, const_basis_indices))
    for (basis_count, const_basis_id) in
        enumerate(Iterators.product(const_basis_indices...))
        basis_indices[basis_count] = lin_num_basis[const_basis_id...]
    end

    return basis_indices
end

function get_support(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    basis_id::Int,
) where {manifold_dim, num_components, num_patches, num_spaces}
    const_support = get_constituent_support(space, basis_id)
    const_num_elements = Geometry.get_num_elements(get_parametric_geometry(space))
    lin_num_elements = LinearIndices(const_num_elements)
    support = Vector{Int}(undef, prod(length, const_support))
    for (element_count, const_element_id) in enumerate(Iterators.product(const_support...))
        support[element_count] = lin_num_elements[const_element_id...]
    end

    return support
end

function get_max_local_dim(space::TensorProductSpace)
    return prod(get_max_local_dim, get_constituent_spaces(space))
end

function get_extraction_coefficients(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    component_id::Int=1,
) where {manifold_dim, num_components, num_patches, num_spaces}
    # The permutations of the constituent spaces should be combined if we allow for more
    # than one component.
    extraction_per_space = get_constituent_extraction(space, element_id)
    extraction_coeffs = kron(
        (extraction_per_space[space][1] for space in num_spaces:-1:1)...
    )

    return extraction_coeffs
end

function get_extraction(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    component_id::Int=1,
) where {manifold_dim, num_components, num_patches, num_spaces}
    extraction_coeffs = get_extraction_coefficients(space, element_id, component_id)

    return extraction_coeffs, 1:size(extraction_coeffs, 2)
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
    const_local_basis = get_constituent_local_basis(space, element_id, xi, nderivatives)
    const_sizes = ntuple(space -> size(const_local_basis[space][1][1][1]), num_spaces)
    num_points = Points.get_num_points(xi)
    num_basis = get_num_basis(space, element_id)
    local_basis = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
    for der_order in 0:nderivatives
        num_der_ids = binomial(manifold_dim + der_order - 1, manifold_dim - 1)
        # We assume that there is only one component.
        local_basis[der_order + 1] = [
            [Matrix{Float64}(undef, (num_points, num_basis))] for _ in 1:num_der_ids
        ]
    end

    der_keys = integer_sums(nderivatives, manifold_dim + 1)
    const_manifold_indices = get_constituent_manifold_indices(space)
    const_eval = ntuple(space -> zeros(const_sizes[space]), num_spaces)
    for key in der_keys
        key = key[1:manifold_dim]
        der_order = sum(key)
        der_id = get_derivative_idx(key)
        for space in 1:num_spaces
            space_der_order = sum(key[const_manifold_indices[space]])
            space_der_id = get_derivative_idx(key[const_manifold_indices[space]])
            const_eval[space] .= const_local_basis[space][space_der_order + 1][space_der_id][1]
        end

        if manifold_dim == 1
            local_basis[der_order + 1][der_id][1] = const_eval[1]
        else
            local_basis[der_order + 1][der_id][1] = kron(
                (const_eval[space] for space in num_spaces:-1:1)...
            )
        end
    end

    return local_basis
end

function evaluate(
    space::TensorProductSpace{manifold_dim, num_components, num_patches, num_spaces},
    element_id::Int,
    xi::Points.CartesianPoints{manifold_dim},
    nderivatives::Int=0,
) where {manifold_dim, num_components, num_patches, num_spaces}
    basis_indices = get_basis_indices(space, element_id)
    num_basis = length(basis_indices)
    const_eval = get_constituent_evaluations(space, element_id, xi, nderivatives)
    num_points = Points.get_num_points(xi)
    eval = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
    for der_order in 0:nderivatives
        num_der_ids = binomial(manifold_dim + der_order - 1, manifold_dim - 1)
        # We assume that there is only one component.
        eval[der_order + 1] = [
            [Matrix{Float64}(undef, (num_points, num_basis))] for _ in 1:num_der_ids
        ]
    end

    const_manifold_indices = get_constituent_manifold_indices(space)
    der_keys = integer_sums(nderivatives, manifold_dim + 1)
    space_der_order = zeros(Int, num_spaces)
    space_der_id = zeros(Int, num_spaces)
    for key in der_keys
        key = key[1:manifold_dim]
        der_order = sum(key)
        der_id = get_derivative_idx(key)

        for space in 1:num_spaces
            space_der_order[space] = sum(key[const_manifold_indices[space]])
            space_der_id[space] = get_derivative_idx(key[const_manifold_indices[space]])
        end

        if num_spaces == 1
            eval[der_order + 1][der_id][1] = const_eval[1][space_der_order[1] + 1][space_der_id[1]][1]
        else
            eval[der_order + 1][der_id][1] = kron(
                (
                    const_eval[space][space_der_order[space] + 1][space_der_id[space]][1] for
                    space in num_spaces:-1:1
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
    const_eval = get_constituent_evaluations(space, element_id, xi, nderivatives)
    num_points = Points.get_num_points(xi)
    eval = Vector{Vector{Vector{Matrix{Float64}}}}(undef, nderivatives + 1)
    for der_order in 0:nderivatives
        num_der_ids = binomial(manifold_dim + der_order - 1, manifold_dim - 1)
        # We assume that there is only one component.
        eval[der_order + 1] = [
            [Matrix{Float64}(undef, (num_points, num_basis))] for _ in 1:num_der_ids
        ]
    end

    const_manifold_indices = get_constituent_manifold_indices(space)
    der_keys = integer_sums(nderivatives, manifold_dim + 1)
    space_der_order = zeros(Int, num_spaces)
    space_der_id = zeros(Int, num_spaces)
    for key in der_keys
        key = key[1:manifold_dim]
        der_order = sum(key)
        der_id = get_derivative_idx(key)

        for space in 1:num_spaces
            space_der_order[space] = sum(key[const_manifold_indices[space]])
            space_der_id[space] = get_derivative_idx(key[const_manifold_indices[space]])
        end

        if num_spaces == 1
            eval[der_order + 1][der_id][1] = const_eval[1][space_der_order[1] + 1][space_der_id[1]][1]
        else
            for point in axes(eval[der_order + 1][der_id][1], 1)
                eval[der_order + 1][der_id][1][point, :] = kron(
                    (
                        const_eval[space][space_der_order[space] + 1][space_der_id[space]][1][
                            point, :,
                        ] for space in num_spaces:-1:1
                    )...,
                )
            end
        end
    end

    return eval, basis_indices
end

# Methods for tensor product B-spline spaces
include("BSplines.jl")
