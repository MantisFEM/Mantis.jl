
include("Basis.jl")
include("Extraction.jl")

############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    HierarchicalBasis{L, H, S, B}

An active hierarchical basis: a hierarchy of `L` level-wise function spaces, together with
the algorithm used to select the active basis functions and the hierarchical basis
representation.

# Fields
- `hierarchy::H`: Hierarchy of level-wise finite element spaces and active basis functions.
- `selection_type::Type{S}`: Algorithm used to construct the active basis.
- `basis_type::Type{B}`: Hierarchical basis representation, such as `HB` or `THB`.

# Outer Constructors
- `HierarchicalBasis(geometry::Geometry.HierarchicalGeometry, scalings, ::Type{S},
  ::Type{B})`: Selects the active basis of `scalings` on `geometry` with `S`.
"""
struct HierarchicalBasis{
    L, H <: Hierarchical.Hierarchy{L}, S <: SelectionAlgorithm, B <: BasisType
}
    hierarchy::H
    selection_type::Type{S}
    basis_type::Type{B}
end

function HierarchicalBasis(
    geometry::Geometry.HierarchicalGeometry, scalings, ::Type{S}, ::Type{B}
) where {S <: SelectionAlgorithm, B <: BasisType}
    hierarchy = create_basis(geometry, scalings, S)

    return HierarchicalBasis(hierarchy, S, B)
end

"""
    get_hierarchy(basis::HierarchicalBasis)

Return the hierarchy of level-wise spaces and active basis functions of `basis`.
"""
get_hierarchy(basis::HierarchicalBasis) = basis.hierarchy
get_selection_type(basis::HierarchicalBasis) = basis.selection_type
get_basis_type(basis::HierarchicalBasis) = basis.basis_type
get_num_levels(basis::HierarchicalBasis) = Hierarchical.get_num_levels(get_hierarchy(basis))

"""
    HierarchicalSpace{manifold_dim, num_components, num_patches, G, GP, HB, E} <:
        AbstractFESpace{manifold_dim, num_components, num_patches}

A finite element space formed from a hierarchy of function spaces and an active hierarchical
basis.

# Fields
- `geometry::G`: Physical geometry of the space.
- `parametric_geometry::GP`: Hierarchical geometry used to index elements and evaluate the
  basis.
- `basis::HB`: Active hierarchical basis. See [`HierarchicalBasis`](@ref).
- `extraction_op::E`: Element-wise extraction coefficients and basis indices.
"""
struct HierarchicalSpace{manifold_dim, num_components, num_patches, G, GP, HB, E} <:
       AbstractFESpace{manifold_dim, num_components, num_patches}
    geometry::G
    parametric_geometry::GP
    basis::HB
    extraction_op::E

    function HierarchicalSpace(
        geometry::G,
        parametric_geometry::GP,
        basis::HB,
        E::ExtractionOperator,
    ) where {
        manifold_dim,
        G <: Geometry.AbstractGeometry{manifold_dim},
        GP <: Geometry.HierarchicalGeometry{manifold_dim},
        HB <: HierarchicalBasis,
    }
        spaces = Hierarchical.get_sets(get_hierarchy(basis))
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

        _check_levels(parametric_geometry, spaces)

        num_components = get_num_components(first(spaces))
        num_patches = get_num_patches(first(spaces))

        return new{manifold_dim, num_components, num_patches, G, GP, HB, typeof(E)}(
            geometry, parametric_geometry, basis, E
        )
    end
end

function HierarchicalSpace(
    geometry::Geometry.AbstractGeometry{manifold_dim},
    parametric_geometry::Geometry.HierarchicalGeometry{manifold_dim},
    basis::HierarchicalBasis,
) where {manifold_dim}
    E = build_extraction_operator(
        parametric_geometry, get_hierarchy(basis), get_basis_type(basis)
    )

    return HierarchicalSpace(geometry, parametric_geometry, basis, E)
end

function HierarchicalSpace(
    geometry,
    parametric_geometry,
    hierarchy::Hierarchical.Hierarchy,
    ::Type{S},
    ::Type{B},
    E::ExtractionOperator,
) where {S <: SelectionAlgorithm, B <: BasisType}
    basis = HierarchicalBasis(hierarchy, S, B)

    return HierarchicalSpace(geometry, parametric_geometry, basis, E)
end

function HierarchicalSpace(
    geometry, parametric_geometry, hierarchy::Hierarchical.Hierarchy, ::Type{S}, ::Type{B}
) where {S <: SelectionAlgorithm, B <: BasisType}
    basis = HierarchicalBasis(hierarchy, S, B)

    return HierarchicalSpace(geometry, parametric_geometry, basis)
end

function HierarchicalSpace(
    geometry, parametric_geometry, scalings, ::Type{S}, ::Type{B}
) where {S <: SelectionAlgorithm, B <: BasisType}
    # The active basis is determined by the active elements, so we use the parametric
    # geometry that indexes them.
    basis = HierarchicalBasis(parametric_geometry, scalings, S, B)

    return HierarchicalSpace(geometry, parametric_geometry, basis)
end

"""
    HierarchicalSpace(
        parametric_geometry::Geometry.HierarchicalGeometry,
        scalings,
        ::Type{S},
        ::Type{B},
    ) where {S <: SelectionAlgorithm, B <: BasisType}

Construct a hierarchical space whose physical geometry is its `parametric_geometry`.
"""
function HierarchicalSpace(
    parametric_geometry::Geometry.HierarchicalGeometry, scalings, ::Type{S}, ::Type{B}
) where {S <: SelectionAlgorithm, B <: BasisType}
    return HierarchicalSpace(parametric_geometry, parametric_geometry, scalings, S, B)
end

"""
    get_hierarchical_basis(space::HierarchicalSpace)

Return the active hierarchical basis of `space`. See [`HierarchicalBasis`](@ref).
"""
get_hierarchical_basis(space::HierarchicalSpace) = space.basis

"""
    get_basis(space::HierarchicalSpace)

Return the hierarchy of level-wise spaces and active basis functions of `space`.
"""
get_basis(space::HierarchicalSpace) = get_hierarchy(get_hierarchical_basis(space))
get_selection_type(space::HierarchicalSpace) =
    get_selection_type(get_hierarchical_basis(space))
get_basis_type(space::HierarchicalSpace) = get_basis_type(get_hierarchical_basis(space))
get_num_levels(space::HierarchicalSpace) = get_num_levels(get_hierarchical_basis(space))

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

function get_support(space::HierarchicalSpace, basis_id::Int)
    return get_support(space, get_basis_type(space), basis_id)
end

# HB
function get_support(space::HierarchicalSpace, ::Type{HB}, basis_id::Int)
    geometry = get_parametric_geometry(space)
    basis = get_basis(space)
    level, level_id = Hierarchical.convert_to_level_and_level_id(basis, basis_id)
    # Get support on the level
    level_support = get_support(Hierarchical.get_set(basis, level), level_id)
    # active in support + active children of inactive in support
    support = mapreduce(
        e -> Geometry.get_active_descendants(geometry, level, e), vcat, level_support
    )

    return support
end

# THB
"""
    get_support(space::HierarchicalSpace, ::Type{THB}, basis_id::Int)

Return the active elements supporting `basis_id`, excluding regions removed by THB
truncation.
"""
function get_support(space::HierarchicalSpace, ::Type{THB}, basis_id::Int)
    geometry = get_parametric_geometry(space)
    basis = get_basis(space)
    level, level_id = Hierarchical.convert_to_level_and_level_id(basis, basis_id)
    # Get support on the level
    level_support = get_support(Hierarchical.get_set(basis, level), level_id)
    # active in support + active children of inactive in support
    support = mapreduce(
        e -> Geometry.get_active_descendants(geometry, level, e), vcat, level_support
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
        # indices.
        # NOTE: It would be preferable to have a closed form expression to determine which
        # elements are not truncated, but this is not easy to do. This local find is an okay
        # alternative, and likely more performant than some complicated logic recursing into
        # active/inactive children.
        keep_ids[i] = basis_id in get_basis_indices(space, e)
    end

    return support[keep_ids]
end

function get_max_local_dim(space::HierarchicalSpace)
    return maximum(e -> length(get_basis_indices(space, e)), 1:get_num_elements(space))
end

############################################################################################
#                                          Update                                          #
############################################################################################

# TODO: Add a `coarsen` method.

"""
    refine(
        space::HierarchicalSpace{manifold_dim},
        level::Int,
        elements::Vector{Int},
    ) where {manifold_dim}

Refine `elements` at `level` and return the updated space. The default method bisects each
element in every manifold dimension.
"""
function refine(
    space::HierarchicalSpace{manifold_dim}, level::Int, elements::Vector{Int}
) where {manifold_dim}
    num_subdivisons = ntuple(_ -> 2, manifold_dim)
    last_basis = Hierarchical.get_child(last(get_scalings(get_basis(space))))
    basis_scaling = scaling_uniform(last_basis, num_subdivisons)
    # The new level geometry has to be the parametric geometry of the new level space.
    last_geo = Hierarchical.get_child(
        last(get_scalings(Geometry.get_hierarchy(get_parametric_geometry(space))))
    )
    geo_scaling = Geometry.scaling_uniform(
        last_geo,
        get_parametric_geometry(Hierarchical.get_child(basis_scaling)),
        num_subdivisons,
    )

    return refine(space, level, elements, geo_scaling, basis_scaling)
end

"""
    refine(
        space::HierarchicalSpace,
        level::Int,
        elements::Vector{Int},
        geo_scaling::AbstractScaling,
        basis_scaling::AbstractScaling,
    )

Refine selected elements using the supplied geometry and basis scalings. The physical and
parametric geometries must be the same object.
"""
function refine(
    space::HierarchicalSpace,
    level::Int,
    elements::Vector{Int},
    geo_scaling::AbstractScaling,
    basis_scaling::AbstractScaling,
)
    if !(get_parametric_geometry(space) === get_geometry(space))
        throw(
            ArgumentError(
                "The parametric and physical geometries must be the same object in memory."
            ),
        )
    end
    # We first update the hierarchical geometry
    geometry = get_parametric_geometry(space)
    geometry = Geometry.refine(geometry, level, elements, geo_scaling)

    return update(space, geometry, geometry, basis_scaling)
end

"""
    update(
        space::HierarchicalSpace{manifold_dim},
        geometry::Geometry.AbstractGeometry{manifold_dim},
        parametric_geometry::Geometry.HierarchicalGeometry{manifold_dim},
        scaling::AbstractScaling,
    ) where {manifold_dim}

Rebuild `space` against the updated geometry and basis scaling, recomputing its active basis
and extraction data.
"""
function update(
    space::HierarchicalSpace{manifold_dim},
    geometry::Geometry.AbstractGeometry{manifold_dim},
    parametric_geometry::Geometry.HierarchicalGeometry{manifold_dim},
    scaling::AbstractScaling,
) where {manifold_dim}
    curr_L = get_num_levels(space)
    next_L = Geometry.get_num_levels(parametric_geometry)
    curr_scalings = get_scalings(get_basis(space))
    # WARNING: This method is type unstable. It's okay here as we immediately call the
    # `HierarchicalSpace` constructor which is type stable.
    scalings = _update_scalings(curr_scalings, scaling, curr_L, next_L)

    # NOTE: The choice here is to recompute the space entirely, instead of trying to reuse
    # the previous basis/extraction computations.
    # This would be a nice improvement for the future, but this is not currently a
    # performance bottleneck and there are few non-trivial details in achieving this:
    #
    # 1. Basis functions have arbitrarily large supports, so updating the active basis from
    #   the changed elements alone is not obvious. A  "local" search is still needed around
    #   a changed element to know if a basis function that was previously supported only on
    #   said element needs to be dropped.
    #
    # 2. Even if the previous point was easier all the numberings of the basis functions
    #   would have to be updated according to what was dropped.
    #
    # 3. A similar story goes for the extraction coefficients, as basis numberings may have
    #   changed, and dropped basis functions can affect arbitrarily finer levels who would
    #   have to update what parents are supported on those levels.

    return HierarchicalSpace(
        geometry,
        parametric_geometry,
        scalings,
        get_selection_type(space),
        get_basis_type(space),
    )
end

function _update_scalings(
    curr_scalings::NTuple{LS, AbstractScaling},
    scaling::AbstractScaling,
    curr_L::Int,
    next_L::Int,
) where {LS}
    # STABILITY: This method is type unstable and should have a barrier immediately after
    # being called.
    level_diff = next_L - curr_L
    if level_diff == 1
        next_scalings = (curr_scalings..., scaling)
    elseif level_diff == 0
        next_scalings = curr_scalings
    elseif level_diff == -1
        next_scalings = front(curr_scalings)
    else
        return throw(
            ArgumentError(
                LazyString(
                    "The number of levels must differ by at most one level. ",
                    "Got ",
                    next_L,
                    " levels for the updated geometry ",
                    " and ",
                    curr_L,
                    " levels for the current space.",
                ),
            ),
        )
    end

    return next_scalings
end
