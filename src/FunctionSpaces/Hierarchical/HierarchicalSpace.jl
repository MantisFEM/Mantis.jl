
include("Basis.jl")
include("Extraction.jl")

############################################################################################
#                                        Structure                                         #
############################################################################################

struct HierarchicalSpace{manifold_dim, num_components, num_patches, G, GP, H, B, E} <:
       AbstractFESpace{manifold_dim, num_components, num_patches}
    geometry::G
    parametric_geometry::GP
    basis::H
    basis_type::Type{B}
    extraction_op::E

    function HierarchicalSpace(
        geometry::G, parametric_geometry::GP, basis::H, basis_type::Type{B}
    ) where {
        manifold_dim,
        L,
        G <: Geometry.AbstractGeometry{manifold_dim},
        GP <: Geometry.HierarchicalGeometry{manifold_dim},
        H <: Hierarchical.Hierarchy{L},
        B <: BasisType,
    }
        spaces = Hierarchical.get_sets(basis)
        for b in spaces
            if !(isa(b, AbstractFESpace))
                throw(
                    ArgumentError(
                        LazyString(
                            "Hierarchical basis must contain only functions. Got type ",
                            typeof(b),
                            " in ",
                            map(b -> Base.typename(typeof(b)).wrapper, spaces),
                        ),
                    ),
                )
            end
        end

        E = build_extraction_operator(parametric_geometry, basis, B)

        num_components = get_num_components(first(spaces))
        num_patches = get_num_patches(first(spaces))

        return new{manifold_dim, num_components, num_patches, G, GP, H, B, typeof(E)}(
            geometry, parametric_geometry, basis, basis_type, E
        )
    end
end

function HierarchicalSpace(
    geometry, parametric_geometry, scalings, ::Type{S}, ::Type{B}
) where {S <: SelectionAlgorithm, B <: BasisType}
    basis = create_basis(geometry, scalings, S)

    return HierarchicalSpace(geometry, parametric_geometry, basis, B)
end

get_basis(space::HierarchicalSpace) = space.basis
get_basis_type(space::HierarchicalSpace) = space.basis_type
get_num_levels(space::HierarchicalSpace) = Hierarchical.get_num_levels(get_basis(space))

function get_local_basis(
    space::HierarchicalSpace{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
    component_id::Int=1,
) where {manifold_dim}
    geometry = get_parametric_geometry(space)
    level, level_id = Geometry.convert_to_level_and_level_id(geometry, element_id)
    space = Hierarchical.get_set(get_basis(space), level)

    return get_local_basis(space, level_id, xi, nderivatives, component_id)
end

# HB
function get_support(
    space::HierarchicalSpace{manifold_dim, num_components, num_patches, G, GP, H, HB},
    basis_id::Int,
) where {manifold_dim, num_components, num_patches, G, GP, H}
    geometry = get_parametric_geometry(space)
    basis = get_basis(space)
    level, level_id = Hierarchical.convert_to_level_and_level_id(basis, basis_id)
    # Get support on the level
    level_support = get_support(Hierarchical.get_set(basis, level), level_id)
    # active in support + active children of inactive in support
    support = mapreduce(
        e -> Geometry.get_nested_active(geometry, level, e), vcat, level_support
    )

    return support
end

# THB
function get_support(
    space::HierarchicalSpace{manifold_dim, num_components, num_patches, G, GP, H, THB},
    basis_id::Int,
) where {manifold_dim, num_components, num_patches, G, GP, H}
    geometry = get_parametric_geometry(space)
    basis = get_basis(space)
    level, level_id = Hierarchical.convert_to_level_and_level_id(basis, basis_id)
    # Get support on the level
    level_support = get_support(Hierarchical.get_set(basis, level), level_id)
    # active in support + active children of inactive in support
    support = mapreduce(
        e -> Geometry.get_nested_active(geometry, level, e), vcat, level_support
    )
    level == get_num_levels(space) && (return support)
    # Drop children elements where the basis function has been truncated
    geo_hier = Geometry.get_hierarchy(geometry)
    keep_ids = BitVector(undef, length(support))
    for (i, e) in enumerate(support)
        l = Hierarchical.get_level(geo_hier, e)
        # Only interested in children elements
        if l == level
            keep_ids[i] = 1
            continue
        end

        # Only keep child element if the basis function is in the child element's basis
        # indices
        keep_ids[i] = basis_id in get_basis_indices(space, e)
    end

    return support[keep_ids]
end

function get_max_local_dim(space::HierarchicalSpace)
    return maximum(e -> length(get_basis_indices(space, e)), 1:get_num_elements(space))
end
