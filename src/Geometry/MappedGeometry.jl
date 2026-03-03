
abstract type AbstractMapping{manifold_dim, image_dim} end

struct Mapping{manifold_dim, image_dim, M, dM, ddM} <:
       AbstractMapping{manifold_dim, image_dim}
    mapping::M
    dmapping::dM
    ddmapping::ddM

    function Mapping(
        dimensions::NTuple{2, Int}, mapping::M, dmapping::dM, ddmapping::ddM=nothing
    ) where {M <: Function, dM <: Function, ddM <: Union{Nothing, Function}}
        return new{dimensions[1], dimensions[2], M, dM, ddM}(mapping, dmapping, ddmapping)
    end

    function Mapping(
        ::Val{manifold_dim},
        ::Val{image_dim},
        mapping::M,
        dmapping::dM,
        ddmapping::ddM=nothing,
    ) where {
        manifold_dim,
        image_dim,
        M <: Function,
        dM <: Function,
        ddM <: Union{Nothing, Function},
    }
        return new{manifold_dim, image_dim, M, dM, ddM}(mapping, dmapping, ddmapping)
    end
end

"""
    get_manifold_dim(mapping::Mapping{manifold_dim, image_dim})

Returns the dimension of the domain manifold of the mapping.

# Arguments
- `::Mapping{manifold_dim, image_dim}`: The mapping structure.

# Returns
- `::Int`: The dimension of the domain manifold.
"""
function get_manifold_dim(
    ::Mapping{manifold_dim, image_dim}
) where {manifold_dim, image_dim}
    return manifold_dim
end

"""
    get_image_dim(mapping::Mapping{manifold_dim, image_dim})

Returns the dimension of the image manifold of the mapping.

# Arguments
- `::Mapping{manifold_dim, image_dim}`: The mapping structure.

# Returns
- `::Int`: The dimension of the image manifold.
"""
function get_image_dim(::Mapping{manifold_dim, image_dim}) where {manifold_dim, image_dim}
    return image_dim
end

"""
    evaluate(mapping::Mapping, x::Matrix{Float64})

Evaluates the mappping of the points `x` from the parametric space to physical space.

# Arguments
- `mapping::Mapping`: The mapping defining the transformation of the points `x`.
- `x::Matrix{Float64}`: The points in parametric space to be mapped.

# Returns
- `::Matrix{Float64}`: The mapped points in physical space. The size of the matrix is
    `(num_points, image_dim)`, where `num_points` is the number of rows in `x` and
    `image_dim` is the dimension of the mapped points.
"""
function evaluate(mapping::Mapping, x::Matrix{Float64})
    image_dim = get_image_dim(mapping)
    num_points = size(x, 1)
    eval = zeros(num_points, image_dim)

    for point in 1:num_points
        eval[point, :] .= mapping.mapping(view(x, point, :))
    end

    return eval
end

function jacobian(
    mapping::Mapping{manifold_dim, image_dim, M, dM}, x::Matrix{Float64}
) where {manifold_dim, image_dim, M <: Function, dM <: Function}
    num_points = size(x, 1)

    return [
        SMatrix{image_dim, manifold_dim}(mapping.dmapping(view(x, i, :))) for
        i in 1:num_points
    ]
end

function hessian(
    mapping::Mapping{manifold_dim, image_dim, M, dM, ddM}, x::Matrix{Float64}
) where {manifold_dim, image_dim, M <: Function, dM <: Function, ddM <: Function}
    return [
        ntuple(image_dim) do i
            return SMatrix{manifold_dim, manifold_dim}(mapping.ddmapping(view(x, p, :))[i])
        end for p in axes(x, 1)
    ]
end

struct MappedGeometry{manifold_dim, image_dim, num_patches, G, Map} <:
       AbstractGeometry{manifold_dim, image_dim, num_patches}
    geometry::G
    mapping::Map
    num_elements::Int
    num_elements_per_patch::NTuple{num_patches, Int}

    # Constructor for multiple mappings, one per patch, with each parametric geometry being
    # a different object for each patch.
    function MappedGeometry(
        geometry::G, mapping::M
    ) where {
        manifold_dim,
        image_dim_base,
        image_dim,
        num_patches,
        G <: NTuple{num_patches, AbstractGeometry{manifold_dim, image_dim_base, 1}},
        M <: NTuple{num_patches, AbstractMapping{manifold_dim, image_dim}},
    }
        num_elements_per_patch = ntuple(num_patches) do geo_i
            get_num_elements(geometry[geo_i])
        end

        return new{manifold_dim, image_dim, num_patches, G, M}(
            geometry, mapping, sum(num_elements_per_patch), num_elements_per_patch
        )
    end

    # Constructor for multiple mappings, one per patch, with the parametric geometry being
    # only one object. If this is a single patch geometry, this one geometry will be used
    # for every patch. If it is a multi-patch geometry, it must have the same number of
    # patches as the number of mappings.
    function MappedGeometry(
        geometry::G, mapping::M
    ) where {
        manifold_dim,
        image_dim_base,
        image_dim,
        num_patches,
        num_patches_G,
        G <: AbstractGeometry{manifold_dim, image_dim_base, num_patches_G},
        M <: NTuple{num_patches, AbstractMapping{manifold_dim, image_dim}},
    }
        if !(num_patches_G == 1 || num_patches_G == num_patches)
            throw(
                ArgumentError(
                    LazyString(
                        "The underlying geometry must have either one patch or the same ",
                        "number of patches as there are mappings, but the geometry has ",
                        num_patches_G,
                        " patches, while there are ",
                        num_patches,
                        " mappings.",
                    ),
                ),
            )
        end

        if num_patches_G == 1
            num_elements_per_patch = ntuple(num_patches) do i
                return get_num_elements(geometry)
            end
        else
            num_elements_per_patch = get_num_elements_per_patch(geometry)
        end

        return new{manifold_dim, image_dim, num_patches, G, M}(
            geometry, mapping, sum(num_elements_per_patch), num_elements_per_patch
        )
    end

    # Constructor for a single mapping for all patches.
    function MappedGeometry(
        geometry::G, mapping::Map
    ) where {
        manifold_dim,
        image_dim_base,
        image_dim,
        num_patches,
        G <: NTuple{num_patches, AbstractGeometry{manifold_dim, image_dim_base, 1}},
        Map <: AbstractMapping{manifold_dim, image_dim},
    }
        num_elements_per_patch = ntuple(num_patches) do geo_i
            get_num_elements(geometry[geo_i])
        end

        return new{manifold_dim, image_dim, num_patches, G, Map}(
            geometry, mapping, sum(num_elements_per_patch), num_elements_per_patch
        )
    end

    # Constructor for a single mapping for a single patch.
    function MappedGeometry(
        geometry::G, mapping::Map
    ) where {
        manifold_dim,
        image_dim_base,
        image_dim,
        G <: AbstractGeometry{manifold_dim, image_dim_base, 1},
        Map <: AbstractMapping{manifold_dim, image_dim},
    }
        num_elements_per_patch = get_num_elements_per_patch(geometry)

        return new{manifold_dim, image_dim, 1, G, Map}(
            geometry, mapping, sum(num_elements_per_patch), num_elements_per_patch
        )
    end
end

# Get properties.
function get_base_geometry(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map}, patch_id::Int=1
) where {manifold_dim, image_dim, num_patches, G <: AbstractGeometry, Map}
    return geometry.geometry
end

function get_base_geometry(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map}, patch_id::Int=1
) where {
    manifold_dim, image_dim, num_patches, G <: NTuple{num_patches, AbstractGeometry}, Map
}
    return geometry.geometry[patch_id]
end

function get_mapping(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map}, patch_id::Int=1
) where {manifold_dim, image_dim, num_patches, G, Map <: Mapping}
    return geometry.mapping
end

function get_mapping(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map}, patch_id::Int=1
) where {manifold_dim, image_dim, num_patches, G, Map <: NTuple{num_patches, Mapping}}
    return geometry.mapping[patch_id]
end

"""
    get_base_patch_and_element_id(geometry::MappedGeometry, element_id::Int)
    get_base_patch_and_element_id(
        geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map}, element_id::Int
    ) where {
        manifold_dim,
        image_dim,
        num_patches,
        G <: AbstractGeometry{manifold_dim, image_dim, num_patches},
        Map,
    }

Compute the `patch_id` and `element_id` for the base geometry, given the `element_id` of
the `MappedGeometry`. Note that the returned `element_id` may still be global, depending on
the type of base geometry.
"""
function get_base_patch_and_element_id(geometry::MappedGeometry, element_id::Int)
    # This function is the default, which assumes that the base geometry is merely a
    # collection of unconnected patches. In that case, we need to compute the patch-local
    # element id to be able to evaluate the base geometry.
    return get_patch_and_local_element_id(geometry, element_id)
end
function get_base_patch_and_element_id(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map}, element_id::Int
) where {
    manifold_dim,
    image_dim,
    num_patches,
    G <: AbstractGeometry{manifold_dim, image_dim, num_patches},
    Map,
}
    # In this case, the base geometry is itself a multi-patch geometry with as many patches
    # as the MappedGeometry. Its elements will thus match, and we don't have to compute a
    # local element id here, since this geometry expects a global element id (and global
    # means the same thing on the base geometry as well as the MappedGeometry).
    base_patch_id = get_patch_id(geometry, element_id)
    return base_patch_id, element_id
end

# Getters for numbers, sizes, shapes, lengths, etc.
function get_element_lengths(geometry::MappedGeometry, element_id::Int)
    base_patch_id, base_element_id = get_base_patch_and_element_id(geometry, element_id)
    return get_element_lengths(get_base_geometry(geometry, base_patch_id), base_element_id)
end

function get_element_measure(geometry::MappedGeometry, element_id::Int)
    base_patch_id, base_element_id = get_base_patch_and_element_id(geometry, element_id)
    return get_element_measure(get_base_geometry(geometry, base_patch_id), base_element_id)
end

function get_element_vertices(geometry::MappedGeometry, element_id::Int)
    base_patch_id, base_element_id = get_base_patch_and_element_id(geometry, element_id)
    return get_element_vertices(get_base_geometry(geometry, base_patch_id), base_element_id)
end

# Evaluations and derivatives.
function evaluate(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, G, Map}
    base_patch_id, base_element_id = get_base_patch_and_element_id(geometry, element_id)
    x = evaluate(get_base_geometry(geometry, base_patch_id), base_element_id, xi)
    x_mapped = evaluate(get_mapping(geometry, base_patch_id), x)

    return x_mapped
end

function jacobian(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches, G, Map}
    # the Jacobian for the mapping from the elements to base geometry image
    base_patch_id, base_element_id = get_base_patch_and_element_id(geometry, element_id)
    base_geometry = get_base_geometry(geometry, base_patch_id)
    J_1 = jacobian(base_geometry, base_element_id, xi)
    x = evaluate(base_geometry, base_element_id, xi)
    # the mapping from the image of the  base geometry to the image of the mapping
    J_2 = jacobian(get_mapping(geometry, base_patch_id), x)

    return J_2 .* J_1
end

function hessian(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    # Jacobian and Hessian of the base geometry
    base_patch_id, base_element_id = get_base_patch_and_element_id(geometry, element_id)
    base_geometry = get_base_geometry(geometry, base_patch_id)
    x = evaluate(base_geometry, base_element_id, xi)
    Jb = jacobian(base_geometry, base_element_id, xi)
    Hb = hessian(base_geometry, base_element_id, xi)

    Jm = jacobian(get_mapping(geometry, base_patch_id), x)
    Hm = hessian(get_mapping(geometry, base_patch_id), x)

    return [
        ntuple(image_dim) do i
            Hp = transpose(Jb[p]) * Hm[p][i] * Jb[p]
            for j in 1:manifold_dim
                Hp += Jm[p][i, j] * Hb[p][j]
            end
            return Hp
        end for p in eachindex(Jb, Jm, Hm, Hb)
    ]
end
