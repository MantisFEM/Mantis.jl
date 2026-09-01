"""
	TensorProductGeometry{
		manifold_dim, image_dim, num_patches, num_geometries, TP
	} <: AbstractGeometry{manifold_dim, image_dim, num_patches}

A geometry built by globally tensoring multiple factor geometries. The resulting
tensor-product geometry has a `manifold_dim` equal to the sum of the factor geometries'
manifold dimensions.

The interface with `TensorProducts` is done by defining the number of objects of a geometry
as the number of elements; the ids then refer to element ids. See [`TensorProducts`](@ref).

# Fields
- `tensor_product::TP`: A `TensorProducts.TensorProduct` of the factor geometries.
- `num_elements_per_patch::NTuple{num_patches, Int}`: The number of elements on each patch.
"""
struct TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries, TP} <:
       AbstractGeometry{manifold_dim, image_dim, num_patches}
    tensor_product::TP
    num_elements_per_patch::NTuple{num_patches, Int}

    function TensorProductGeometry(
        geometries::T
    ) where {num_geometries, T <: NTuple{num_geometries, AbstractGeometry}}
        tensor_product = TensorProduct(geometries)
        manifold_dim = sum(get_manifold_dim, geometries)
        image_dim = sum(get_image_dim, geometries)
        num_patches = prod(get_num_patches, geometries)

        # The iteration over elements also works in the multi-patch case due to the (global)
        # tensor-product structure. However, to make it compatible with other multi -patch
        # functions, an efficient get_num_elements_per_patch is needed. This is easily and
        # efficiently done once here and then stored.
        factor_num_patches = map(get_num_patches, geometries)
        cart_num_patches = CartesianIndices(factor_num_patches)
        num_elements_per_patch = ntuple(num_patches) do patch_id
            return prod(
                map(get_num_elements, geometries, Tuple(cart_num_patches[patch_id]))
            )
        end

        return new{
            manifold_dim, image_dim, num_patches, num_geometries, typeof(tensor_product)
        }(
            tensor_product, num_elements_per_patch
        )
    end
end

TensorProductGeometry(geometries...) = TensorProductGeometry(geometries)

get_tensor_product(geometry::TensorProductGeometry) = geometry.tensor_product

function get_num_elements(geometry::TensorProductGeometry)
    #=
    The `TensorProducts` module already implements a `get_num_objects` by checking the size
    of the `CartesianIndices` iterator.
    However, `TensorProducts` needs to call `get_num_objects` on the factor geometries, which
    is why there is also a `TensorProducts.get_num_objects(geometry::AbstractGeometry)`.
    =#
    return TensorProducts.get_num_objects(get_tensor_product(geometry))
end

"""
    get_cart_num_elements(geometry::TensorProductGeometry)

Return a `CartesianIndices` iterator over the number of elements in `geometry`.
"""
function get_cart_num_elements(geometry::TensorProductGeometry)
    return TensorProducts.get_cart_ids(get_tensor_product(geometry))
end

"""
    get_lin_num_elements(geometry::TensorProductGeometry)

Return a `LinearIndices` iterator over the number of elements in `geometry`.
"""
function get_lin_num_elements(geometry::TensorProductGeometry)
    return TensorProducts.get_lin_ids(get_tensor_product(geometry))
end

function get_num_geometries(
    ::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries}
) where {manifold_dim, image_dim, num_patches, num_geometries}
    return num_geometries
end

TensorProducts.get_factors(geometry::TensorProductGeometry) = get_factor_geometries(geometry)

function TensorProducts.get_factor_ids(geometry::TensorProductGeometry, element_id::Int)
    # first to remove patch id
    return first(get_factor_element_ids(geometry, element_id))
end

TensorProducts.get_lin_ids(geometry::TensorProductGeometry) = get_lin_num_elements(geometry)

"""
    get_factor_geometries(geometry::TensorProductGeometry)

Return a tuple with all the factor geometries tensored in `geometry`.
"""
function get_factor_geometries(geometry::TensorProductGeometry)
    return TensorProducts.get_factors(get_tensor_product(geometry))
end

"""
    get_factor_num_elements(geometry::TensorProductGeometry)

Return the number of elements in the factor geometries.
"""
function get_factor_num_elements(geometry::TensorProductGeometry)
    return TensorProducts.get_factor_num_objects(get_tensor_product(geometry))
end

function get_factor_element_ids(geometry::TensorProductGeometry, element_id::Int)
    # While the elements are globally tensored in TPGeometry, the output should be
    # consistent with the same function for CartesianGeometry, which also returns the
    # patch_id, so we do that here too.
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)

    return TensorProducts.get_factor_ids(get_tensor_product(geometry), local_element_id),
    patch_id
end

"""
    get_factor_manifold_dims(geometry::TensorProductGeometry)

Return the manifold dimension of the factor geometries.
"""
function get_factor_manifold_dims(geometry::TensorProductGeometry)
    return TensorProducts.mapfactors(get_manifold_dim, get_tensor_product(geometry))
end

"""
    get_factor_image_dims(geometry::TensorProductGeometry)

Return the image dimension of the factor geometries.
"""
function get_factor_image_dims(geometry::TensorProductGeometry)
    return TensorProducts.mapfactors(get_image_dim, get_tensor_product(geometry))
end

"""
    get_factor_num_patches(geometry::TensorProductGeometry)

Return the number of patches of the factor geometries.
"""
function get_factor_num_patches(geometry::TensorProductGeometry)
    return TensorProducts.mapfactors(get_num_patches, get_tensor_product(geometry))
end

function get_factor_manifold_indices(geometry::TensorProductGeometry)
    factor_manifold_dims = get_factor_manifold_dims(geometry)
    cum_factor_manifold_dim = (0, cumsum(factor_manifold_dims)...)
    factor_manifold_indices = ntuple(
        geometry -> ntuple(
            i -> cum_factor_manifold_dim[geometry] + i, factor_manifold_dims[geometry]
        ),
        get_num_geometries(geometry),
    )

    return factor_manifold_indices
end

"""
    get_factor_image_indices(geometry::TensorProductGeometry)

See [`get_factor_manifold_indices`](@ref).
"""
function get_factor_image_indices(geometry::TensorProductGeometry)
    factor_image_dims = get_factor_image_dims(geometry)
    cum_factor_image_dim = (0, cumsum(factor_image_dims)...)
    factor_image_indices = ntuple(
        geometry ->
            ntuple(i -> cum_factor_image_dim[geometry] + i, factor_image_dims[geometry]),
        get_num_geometries(geometry),
    )

    return factor_image_indices
end

function get_factor_element_vertices(geometry::TensorProductGeometry, element_id::Int)
    return TensorProducts.mapfactors(
        get_element_vertices, get_tensor_product(geometry), element_id
    )
end

function get_factor_element_lengths(geometry::TensorProductGeometry, element_id::Int)
    return TensorProducts.mapfactors(
        get_element_lengths, get_tensor_product(geometry), element_id
    )
end

function get_factor_evaluation_points(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    factor_manifold_indices = get_factor_manifold_indices(geometry)
    input_points = Points.get_input_points(xi)
    factor_xi = ntuple(num_geometries) do geo
        factor_indices = factor_manifold_indices[geo]
        factor_range = factor_indices[1]:factor_indices[end]

        return Points.TensorProductPoints(input_points[factor_range])
    end

    return factor_xi
end

function get_factor_evaluations(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    factor_geometries = get_factor_geometries(geometry)
    factor_element_ids, _ = get_factor_element_ids(geometry, element_id)
    factor_xi = get_factor_evaluation_points(geometry, xi)

    return map(evaluate, factor_geometries, factor_element_ids, factor_xi)
end

function get_factor_jacobians(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    factor_geometries = get_factor_geometries(geometry)
    factor_element_ids, _ = get_factor_element_ids(geometry, element_id)
    factor_xi = get_factor_evaluation_points(geometry, xi)

    return map(jacobian, factor_geometries, factor_element_ids, factor_xi)
end

function get_factor_hessians(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    factor_geometries = get_factor_geometries(geometry)
    factor_element_ids, _ = get_factor_element_ids(geometry, element_id)
    factor_xi = get_factor_evaluation_points(geometry, xi)

    return map(hessian, factor_geometries, factor_element_ids, factor_xi)
end

function get_element_vertices(geometry::TensorProductGeometry, element_id::Int)
    factor_element_vertices = get_factor_element_vertices(geometry, element_id)
    factor_manifold_dim = get_factor_manifold_dims(geometry)
    cum_factor_manifold_dim = (0, cumsum(factor_manifold_dim)...)
    element_vertices = ntuple(get_manifold_dim(geometry)) do dim
        factor_geometry_id = findfirst(
            cum_manifold_dim -> dim ≤ cum_manifold_dim, cum_factor_manifold_dim[2:end]
        )
        factor_dim_id = dim - cum_factor_manifold_dim[factor_geometry_id]

        return factor_element_vertices[factor_geometry_id][factor_dim_id]
    end

    return element_vertices
end

function get_element_lengths(geometry::TensorProductGeometry, element_id::Int)
    factor_element_lengths = get_factor_element_lengths(geometry, element_id)
    factor_manifold_dim = get_factor_manifold_dims(geometry)
    cum_factor_manifold_dim = (0, cumsum(factor_manifold_dim)...)
    element_lengths = ntuple(Val(get_manifold_dim(geometry))) do dim
        factor_geometry_id = findfirst(
            cum_manifold_dim -> dim ≤ cum_manifold_dim, cum_factor_manifold_dim[2:end]
        )
        factor_dim_id = dim - cum_factor_manifold_dim[factor_geometry_id]

        return factor_element_lengths[factor_geometry_id][factor_dim_id]
    end

    return element_lengths
end

function get_element_measure(geometry::TensorProductGeometry, element_id::Int)
    return prod(get_element_lengths(geometry, element_id))
end

# Evaluations and derivatives.
function evaluate(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.TensorProductPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    factor_evaluations = get_factor_evaluations(geometry, element_id, xi)
    factor_eval_points = get_factor_evaluation_points(geometry, xi)
    num_points = Points.get_num_points(xi)
    eval = zeros(num_points, image_dim)
    factor_image_indices = get_factor_image_indices(geometry)
    factor_num_points = map(Points.get_num_points, factor_eval_points)
    cart_num_points = CartesianIndices(factor_num_points)
    for geo_id in 1:num_geometries
        factor_image_range = factor_image_indices[geo_id][1]:factor_image_indices[geo_id][end]
        for point in axes(eval, 1)
            eval[point, factor_image_range] .= @view factor_evaluations[geo_id][
                cart_num_points[point][geo_id], :,
            ]
        end
    end

    return eval
end

function evaluate(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    factor_evaluations = get_factor_evaluations(geometry, element_id, xi)
    num_points = Points.get_num_points(xi)
    eval = zeros(num_points, image_dim)
    factor_image_indices = get_factor_image_indices(geometry)
    for geo_id in 1:num_geometries
        factor_image_range = factor_image_indices[geo_id][1]:factor_image_indices[geo_id][end]
        for point in axes(eval, 1)
            eval[point, factor_image_range] .= @view factor_evaluations[geo_id][point, :]
        end
    end

    return eval
end

function jacobian(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.TensorProductPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    factor_jacobians = get_factor_jacobians(geometry, element_id, xi)
    factor_image_indices = get_factor_image_indices(geometry)
    factor_manifold_indices = get_factor_manifold_indices(geometry)
    factor_eval_points = get_factor_evaluation_points(geometry, xi)
    factor_num_points = map(Points.get_num_points, factor_eval_points)
    cart_num_points = CartesianIndices(factor_num_points)

    num_eval_points = Points.get_num_points(xi)
    J = Vector{SMatrix{image_dim, manifold_dim, Float64, image_dim * manifold_dim}}(
        undef, num_eval_points
    )
    Jp = Ref(zero(MMatrix{image_dim, manifold_dim, Float64, image_dim * manifold_dim}))
    geo_id = Ref(1)
    for point in eachindex(J)
        jacs_per_point = map(getindex, factor_jacobians, Tuple(cart_num_points[point]))
        foreach(jacs_per_point) do jac_i
            factor_manifold_range =
                factor_manifold_indices[geo_id[]][1]:factor_manifold_indices[geo_id[]][end]
            factor_image_range =
                factor_image_indices[geo_id[]][1]:factor_image_indices[geo_id[]][end]
            setindex!(Jp[], jac_i, factor_image_range, factor_manifold_range)
            return geo_id[] += 1
        end
        J[point] = SMatrix{image_dim, manifold_dim}(Jp[])
        geo_id[] = 1
    end
    return J
end

function jacobian(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    factor_jacobians = get_factor_jacobians(geometry, element_id, xi)
    factor_image_indices = get_factor_image_indices(geometry)
    factor_manifold_indices = get_factor_manifold_indices(geometry)

    num_eval_points = Points.get_num_points(xi)
    J = Vector{SMatrix{image_dim, manifold_dim, Float64, image_dim * manifold_dim}}(
        undef, num_eval_points
    )
    Jp = Ref(zero(MMatrix{image_dim, manifold_dim, Float64, image_dim * manifold_dim}))
    geo_id = Ref(1)
    for point in eachindex(J)
        jacs_per_point = map(getindex, factor_jacobians, ntuple(i -> i, num_geometries))
        foreach(jacs_per_point) do jac_i
            factor_manifold_range =
                factor_manifold_indices[geo_id[]][1]:factor_manifold_indices[geo_id[]][end]
            factor_image_range =
                factor_image_indices[geo_id[]][1]:factor_image_indices[geo_id[]][end]
            setindex!(Jp[], jac_i, factor_image_range, factor_manifold_range)
            return geo_id[] += 1
        end
        J[point] = SMatrix{image_dim, manifold_dim}(Jp[])
        geo_id[] = 1
    end

    return J
end

function hessian(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.TensorProductPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    factor_hessians = get_factor_hessians(geometry, element_id, xi)
    factor_image_indices = get_factor_image_indices(geometry)
    factor_manifold_indices = get_factor_manifold_indices(geometry)
    factor_eval_points = get_factor_evaluation_points(geometry, xi)
    factor_num_points = map(Points.get_num_points, factor_eval_points)
    cart_num_points = CartesianIndices(factor_num_points)

    num_eval_points = Points.get_num_points(xi)

    H = [
        _hessian_per_point(
            geometry,
            factor_hessians,
            factor_image_indices,
            factor_manifold_indices,
            Tuple(cart_num_points[point]),
        ) for point in 1:num_eval_points
    ]

    return H
end

function hessian(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    factor_hessians = get_factor_hessians(geometry, element_id, xi)
    factor_image_indices = get_factor_image_indices(geometry)
    factor_manifold_indices = get_factor_manifold_indices(geometry)

    num_eval_points = Points.get_num_points(xi)

    H = [
        _hessian_per_point(
            geometry,
            factor_hessians,
            factor_image_indices,
            factor_manifold_indices,
            ntuple(i -> i, num_geometries),
        ) for _ in 1:num_eval_points
    ]

    return H
end

function _hessian_per_point(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    factor_hessians,
    factor_image_indices,
    factor_manifold_indices,
    tup,
) where {manifold_dim, image_dim, num_patches, num_geometries}
    Hp = Ref(
        zero(MMatrix{manifold_dim, manifold_dim, Float64, manifold_dim * manifold_dim})
    )
    geo_id = Ref(1)

    factor_hessians_per_point = map(getindex, factor_hessians, tup)

    return ntuple(Val(image_dim)) do im_i
        Hp[] = zero(
            MMatrix{manifold_dim, manifold_dim, Float64, manifold_dim * manifold_dim}
        )

        foreach(factor_hessians_per_point) do hessian_i
            factor_im_i = findfirst(
                isequal(im_i),
                factor_image_indices[geo_id[]][1]:factor_image_indices[geo_id[]][end],
            )
            factor_manifold_range =
                factor_manifold_indices[geo_id[]][1]:factor_manifold_indices[geo_id[]][end]
            if !isnothing(factor_im_i)
                setindex!(
                    Hp[], hessian_i[factor_im_i], factor_manifold_range, factor_manifold_range
                )
            end
            return geo_id[] += 1
        end
        geo_id[] = 1

        return SMatrix{manifold_dim, manifold_dim}(Hp[])
    end
end
