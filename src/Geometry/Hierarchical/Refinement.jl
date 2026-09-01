############################################################################################
#                                         Uniform                                          #
############################################################################################

function refinement_uniform(
    parent_geo::AbstractGeometry{manifold_dim}, num_subdivisons::NTuple{manifold_dim, Int}
) where {manifold_dim}
    return throw(MethodError(refinement_uniform, (parent_geo, num_subdivisons)))
end

function refinement_uniform(
    parent_geo::AbstractGeometry{manifold_dim}, num_subdivisons::Int
) where {manifold_dim}
    return refinement_uniform(parent_geo, ntuple(_ -> num_subdivisons, manifold_dim))
end

function refinement_uniform(
    parent_geo::Geometry.UnstructuredGeometry{manifold_dim, image_dim, num_patches},
    num_subdivisions::NTuple{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    return refinement_uniform(parent_geo, ntuple(_ -> num_subdivisions, num_patches))
end

function refinement_uniform(
    parent_geo::Geometry.UnstructuredGeometry{manifold_dim, image_dim, num_patches},
    num_subdivisions::NTuple{num_patches, NTuple{manifold_dim}},
) where {manifold_dim, image_dim, num_patches}
    patch_parent_geo = Geometry.get_geometry_per_patch(parent_geo)
    patch_child_geo = ntuple(
        patch -> refinement_uniform(patch_parent_geo[patch], num_subdivisions[patch]),
        num_patches,
    )
    child_geo = Geometry.UnstructuredGeometry(patch_child_geo)

    return child_geo
end

function refinement_uniform(parent_geo::Geometry.MappedGeometry, num_subdivisons)
    parent_base_geometry = Geometry.get_base_geometry(parent_geo)
    child_base_geometry = refinement_uniform(parent_base_geometry, num_subdivisons)
    mapping = Geometry.get_mapping(parent_geo)
    child_geo = Geometry.MappedGeometry(child_base_geometry, mapping)

    return child_geo
end

function refinement_uniform(
    parent_geo::Geometry.TensorProductGeometry{
        manifold_dim, image_dim, num_patches, num_geometries
    },
    num_subdivisons::NTuple{manifold_dim},
) where {manifold_dim, image_dim, num_patches, num_geometries}
    factor_parent_geo = Geometry.get_factor_geometries(parent_geo)
    factor_manifold_indices = Geometry.get_factor_manifold_indices(parent_geo)
    factor_num_subdivions = ntuple(
        geo ->
            num_subdivisons[factor_manifold_indices[geo][1]:factor_manifold_indices[geo][end]],
        num_geometries,
    )
    factor_child_geo = ntuple(
        geo -> refinement_uniform(factor_parent_geo[geo], factor_num_subdivions[geo]),
        num_geometries,
    )
    child_geo = Geometry.TensorProductGeometry(factor_child_geo)

    return child_geo
end

function refinement_uniform(
    parent_geo::Geometry.CartesianGeometry{manifold_dim},
    num_subdivisons::NTuple{manifold_dim, Int},
) where {manifold_dim}
    parent_breakpoints = Geometry.get_breakpoints(parent_geo)
    factor_child_breakpoints = ntuple(
        dim ->
            refinement_uniform_breakpoints(parent_breakpoints[dim], num_subdivisons[dim]),
        manifold_dim,
    )
    child_geo = Geometry.CartesianGeometry(factor_child_breakpoints)

    return child_geo
end

function refinement_uniform(
    parent_breakpoints::AbstractVector{<:Real}, num_subdivisions; T=Float64
)
    if isempty(parent_breakpoints)
        throw(ArgumentError("`parent_breakpoints` is empty."))
    end

    num_parent_breakpoints = length(parent_breakpoints)
    parent_num_elements = num_parent_breakpoints - 1
    num_child_breakpoints = parent_num_elements * num_subdivisions + 1
    child_breakpoints = Vector{T}(undef, num_child_breakpoints)
    child_breakpoints[end] = parent_breakpoints[end]
    k = 1
    @inbounds for i in 1:parent_num_elements
        a = parent_breakpoints[i]
        b = parent_breakpoints[i + 1]
        step = (b - a) / num_subdivisions
        for j in 0:(num_subdivisions - 1)
            child_breakpoints[k] = a + j * step
            k += 1
        end
    end

    return child_breakpoints
end

function refinement_uniform_breakpoints(parent_breakpoints::LinRange, num_subdivisions)
    start_breakpoint = first(parent_breakpoints)
    end_breakpoint = last(parent_breakpoints)
    parent_num_elements = length(parent_breakpoints) - 1
    child_breakpoints = LinRange(
        start_breakpoint, end_breakpoint, parent_num_elements * num_subdivisions + 1
    )

    return child_breakpoints
end
