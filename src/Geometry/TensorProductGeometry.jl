"""
    TensorProductGeometry{manifold_dim, image_dim, num_geometries, T, CI, LI} <:
    AbstractGeometry{manifold_dim}

A geometry type representing the tensor product of multiple constituent geometries.

# Fields
- `geometries::T`: Tuple of constituent geometries.
- `cart_num_elements::CI`: A collection of Cartesian indices representing the
  multi-dimensional positions of each tensor product element. Useful to convert from linear
  to Cartesian indexing.
- `lin_num_elements::LI`: Useful to convert from Cartesian to linear indexing.
"""
struct TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries, T, CI, LI} <:
       AbstractGeometry{manifold_dim, image_dim, num_patches}
    geometries::T
    cart_num_elements::CI
    lin_num_elements::LI

    function TensorProductGeometry(
        geometries::T
    ) where {num_geometries, T <: NTuple{num_geometries, AbstractGeometry}}
        const_num_elements = ntuple(
            geometry -> get_num_elements(geometries[geometry]), num_geometries
        )
        cart_num_elements = CartesianIndices(const_num_elements)
        lin_num_elements = LinearIndices(cart_num_elements)
        manifold_dim = sum(get_manifold_dim, geometries)
        image_dim = sum(get_image_dim, geometries)

        return new{
            manifold_dim,
            image_dim,
            1,
            num_geometries,
            T,
            typeof(cart_num_elements),
            typeof(lin_num_elements),
        }(
            geometries, cart_num_elements, lin_num_elements
        )
    end
end

get_cart_num_elements(geometry::TensorProductGeometry) = geometry.cart_num_elements
get_lin_num_elements(geometry::TensorProductGeometry) = geometry.lin_num_elements
get_constituent_geometries(geometry::TensorProductGeometry) = geometry.geometries
get_num_geometries(::TensorProductGeometry{
    manifold_dim, image_dim, num_patches, num_geometries, T, CI, LI
}) where{manifold_dim, image_dim, num_patches, num_geometries, T, CI, LI} = num_geometries

function get_num_elements(geometry::TensorProductGeometry)
    return prod(get_constituent_num_elements(geometry))
end

function get_constituent_geometry(geometry::TensorProductGeometry, geometry_id::Int)
    return get_constituent_geometries(geometry)[geometry_id]
end

function get_parametric_geometry(geometry::TensorProductGeometry, patch_id::Int)
    return TensorProductGeometry(
        ntuple(
            geo -> get_parametric_geometry(get_constituent_geometry(geometry, geo)),
            get_num_geometries(geometry),
        )
    )
end

function get_constituent_num_elements(geometry::TensorProductGeometry)
    return Tuple(maximum(get_cart_num_elements(geometry)))
end

function get_constituent_element_id(geometry::TensorProductGeometry, element_id::Int)
    return get_cart_num_elements(geometry)[element_id]
end

function get_constituent_manifold_dim(geometry::TensorProductGeometry)
    const_geometries = get_constituent_geometries(geometry)

    return ntuple(geometry -> get_manifold_dim(const_geometries[geometry]), get_num_geometries(geometry))
end

function get_constituent_image_dim(geometry::TensorProductGeometry)
    const_geometries = get_constituent_geometries(geometry)

    return ntuple(geometry -> get_image_dim(const_geometries[geometry]), get_num_geometries(geometry))
end

function get_constituent_manifold_indices(geometry::TensorProductGeometry)
    const_manifold_dim = get_constituent_manifold_dim(geometry)
    cum_const_manifold_dim = (0, cumsum(const_manifold_dim)...)
    const_manifold_indices = ntuple(
        geometry ->
            (cum_const_manifold_dim[geometry] + 1):cum_const_manifold_dim[geometry + 1],
        get_num_geometries(geometry),
    )

    return const_manifold_indices
end

function get_constituent_image_indices(geometry::TensorProductGeometry)
    const_image_dim = get_constituent_image_dim(geometry)
    cum_const_image_dim = (0, cumsum(const_image_dim)...)
    const_image_indices = ntuple(
        geometry -> (cum_const_image_dim[geometry] + 1):cum_const_image_dim[geometry + 1],
        get_num_geometries(geometry),
    )

    return const_image_indices
end

function get_constituent_element_vertices(geometry::TensorProductGeometry, element_id::Int)
    const_spaces = get_constituent_geometries(geometry)
    const_element_id = get_constituent_element_id(geometry, element_id)
    const_element_vertices = ntuple(
        geometry ->
            get_element_vertices(const_spaces[geometry], const_element_id[geometry]),
        get_num_geometries(geometry),
    )

    return const_element_vertices
end

function get_constituent_element_lengths(geometry::TensorProductGeometry, element_id::Int)
    const_spaces = get_constituent_geometries(geometry)
    const_element_id = get_constituent_element_id(geometry, element_id)
    const_element_lengths = ntuple(
        geometry -> get_element_lengths(const_spaces[geometry], const_element_id[geometry]),
        get_num_geometries(geometry),
    )

    return const_element_lengths
end

function get_element_vertices(geometry::TensorProductGeometry, element_id::Int)
    const_element_vertices = get_constituent_element_vertices(geometry, element_id)
    const_manifold_dim = get_constituent_manifold_dim(geometry)
    cum_const_manifold_dim = (0, cumsum(const_manifold_dim)...)
    element_vertices = ntuple(get_manifold_dim(geometry)) do dim
        const_geometry_id = findfirst(
            cum_manifold_dim -> dim ≤ cum_manifold_dim, cum_const_manifold_dim[2:end]
        )
        const_dim_id = dim - cum_const_manifold_dim[const_geometry_id]

        return const_element_vertices[const_geometry_id][const_dim_id]
    end

    return element_vertices
end

function get_element_lengths(geometry::TensorProductGeometry, element_id::Int)
    const_element_lengths = get_constituent_element_lengths(geometry, element_id)
    const_manifold_dim = get_constituent_manifold_dim(geometry)
    cum_const_manifold_dim = (0, cumsum(const_manifold_dim)...)
    element_lengths = ntuple(get_manifold_dim(geometry)) do dim
        const_geometry_id = findfirst(
            cum_manifold_dim -> dim ≤ cum_manifold_dim, cum_const_manifold_dim[2:end]
        )
        const_dim_id = dim - cum_const_manifold_dim[const_geometry_id]

        return const_element_lengths[const_geometry_id][const_dim_id]
    end

    return element_lengths
end

function get_element_measure(geometry::TensorProductGeometry, element_id::Int)
    return prod(get_element_lengths(geometry, element_id))
end

function get_constituent_evaluation_points(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    const_manifold_indices = get_constituent_manifold_indices(geometry)
    const_points = Points.get_constituent_points(xi)
    const_xi = ntuple(
        geometry -> Points.CartesianPoints(const_points[const_manifold_indices[geometry]]),
        num_geometries,
    )

    return const_xi
end

function get_constituent_evaluations(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    const_geometries = get_constituent_geometries(geometry)
    const_element_id = get_constituent_element_id(geometry, element_id)
    const_xi = get_constituent_evaluation_points(geometry, xi)
    const_eval = ntuple(
        geometry -> evaluate(
            const_geometries[geometry], const_element_id[geometry], const_xi[geometry]
        ),
        num_geometries,
    )

    return const_eval
end

function get_constituent_jacobians(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    const_geometries = get_constituent_geometries(geometry)
    const_element_id = get_constituent_element_id(geometry, element_id)
    const_xi = get_constituent_evaluation_points(geometry, xi)
    const_eval = ntuple(
        geometry -> jacobian(
            const_geometries[geometry], const_element_id[geometry], const_xi[geometry]
        ),
        num_geometries,
    )

    return const_eval
end

"""
    evaluate(
        geometry::TensorProductGeometry{manifold_dim, T},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim}
    ) where {
        manifold_dim, image_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
    }

Go [here](Modules/Geometry.md) for more details.
"""
function evaluate(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.CartesianPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    const_evaluations = get_constituent_evaluations(geometry, element_id, xi)
    num_points = Points.get_num_points(xi)
    eval = zeros(num_points, image_dim)
    const_image_indices = get_constituent_image_indices(geometry)
    cart_num_points = CartesianIndices(
        ntuple(
            geo ->
                Points.get_num_points(get_constituent_evaluation_points(geometry, xi)[geo]),
            num_geometries,
        ),
    )
    for geo_id in 1:num_geometries
        for point in axes(eval, 1)
            eval[point, const_image_indices[geo_id]] .= @view const_evaluations[geo_id][
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
    const_evaluations = get_constituent_evaluations(geometry, element_id, xi)
    num_points = Points.get_num_points(xi)
    eval = zeros(num_points, image_dim)
    const_image_indices = get_constituent_image_indices(geometry)
    for geo_id in 1:num_geometries
        for point in axes(eval, 1)
            eval[point, const_image_indices[geo_id]] .= @view const_evaluations[geo_id][
                point, :,
            ]
        end
    end

    return eval
end

"""
    jacobian(
        geometry::TensorProductGeometry{manifold_dim, T},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim}
    ) where {
        manifold_dim, image_dim, num_geometries, T<:NTuple{num_geometries, AbstractGeometry}
    }

Go [here](Modules/Geometry.md) for more details.
"""
function jacobian(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.CartesianPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    const_jacobians = get_constituent_jacobians(geometry, element_id, xi)
    const_image_indices = get_constituent_image_indices(geometry)
    const_manifold_indices = get_constituent_manifold_indices(geometry)
    cart_num_points = CartesianIndices(
        ntuple(
            geo ->
                Points.get_num_points(get_constituent_evaluation_points(geometry, xi)[geo]),
            num_geometries,
        ),
    )

    num_eval_points = Points.get_num_points(xi)
    J = Vector{SMatrix{image_dim, manifold_dim, Float64}}(undef, num_eval_points)
    for point in 1:num_eval_points
        Jp = zeros(image_dim, manifold_dim)
        for geo_id in 1:num_geometries
            Jp[
                const_image_indices[geo_id], const_manifold_indices[geo_id]
            ] .= @view const_jacobians[geo_id][cart_num_points[point][geo_id]][:, :]
        end
        J[point] = SMatrix{image_dim, manifold_dim}(Jp)
    end

    return J
end

function jacobian(
    geometry::TensorProductGeometry{manifold_dim, image_dim, num_patches, num_geometries},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    const_jacobians = get_constituent_jacobians(geometry, element_id, xi)
    const_image_indices = get_constituent_image_indices(geometry)
    const_manifold_indices = get_constituent_manifold_indices(geometry)

    num_eval_points = Points.get_num_points(xi)
    J = Vector{SMatrix{image_dim, manifold_dim, Float64}}(undef, num_eval_points)
    for point in 1:num_eval_points
        Jp = zeros(image_dim, manifold_dim)
        for geo_id in 1:num_geometries
            Jp[
                const_image_indices[geo_id], const_manifold_indices[geo_id]
            ] .= @view const_jacobians[geo_id][point][:, :]
        end
        J[point] = SMatrix{image_dim, manifold_dim}(Jp)
    end

    return J
end
