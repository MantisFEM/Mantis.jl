
############################################################################################
#                                        Relations                                         #
############################################################################################

"""
    parent_to_children_uniform(
        geometry::AbstractGeometry{manifold_dim}, num_subdivisions::NTuple{manifold_dim, Int}
    ) where {manifold_dim}

Return the parent-to-child relation resulting from subdividing `geometry` uniformly by
`num_subdivions`.

See also [`refinement_uniform`](@ref) and [`Hierarchical.RelationExplicit`](@ref).
"""
function parent_to_children_uniform(
    geometry::AbstractGeometry{manifold_dim}, num_subdivisions::NTuple{manifold_dim, Int}
) where {manifold_dim}
    return throw(MethodError(parent_to_children_uniform, (geometry, num_subdivisions)))
end

"""
    child_to_parents_uniform(
        geometry::AbstractGeometry{manifold_dim}, num_subdivisions::NTuple{manifold_dim, Int}
    ) where {manifold_dim}

Return the child-to-parents relation resulting from subdividing the _parent_ of `geometry`
uniformly by `num_subdivions`.

!!!note
    A geometry has only one parent for a given child, but this is not true for general
    relations. For consistency, the relation method wraps the parent in an indexable type,
    such as a tuple or vector.

See also [`refinement_uniform`](@ref) and [`Hierarchical.RelationExplicit`](@ref).
"""
function child_to_parents_uniform(
    geometry::AbstractGeometry{manifold_dim}, num_subdivisions::NTuple{manifold_dim, Int}
) where {manifold_dim}
    return throw(MethodError(child_to_parents_uniform, (geometry, num_subdivisions)))
end

# 1D Generic

function parent_to_children_uniform(geometry::AbstractGeometry{1}, num_subdivisions::Int)
    return parent_to_children_uniform(num_subdivisions)
end

function child_to_parents_uniform(::AbstractGeometry{1}, num_subdivisions::Int)
    return child_to_parents_uniform(num_subdivisions)
end

function parent_to_children_uniform(
    geometry::AbstractGeometry{manifold_dim}, num_subdivisions::Int
) where {manifold_dim}
    return parent_to_children_uniform(geometry, ntuple(_ -> num_subdivisions, manifold_dim))
end

function child_to_parents_uniform(
    geometry::AbstractGeometry{manifold_dim}, num_subdivisions::Int
) where {manifold_dim}
    return child_to_parents_uniform(geometry, ntuple(_ -> num_subdivisions, manifold_dim))
end

"""
    parent_to_children_uniform(num_subdivisions::Int)

Return the parent-to-child relation resulting from a one-dimensional, uniform subdivision of
elements.
"""
function parent_to_children_uniform(num_subdivisions::Int)
    _check_num_subdivsions(num_subdivisions)

    return RelationExplicit{Hierarchical.PC}(
        e -> _parent_to_children_uniform(e, num_subdivisions)
    )
end

function _parent_to_children_uniform(parent_element_id, num_subdivisions)
    offset = (parent_element_id - 1) * num_subdivisions

    return (offset + 1):(offset + num_subdivisions)
end

"""
    parent_to_children_uniform(num_subdivisions::Int)

Return the child-to-parents relation resulting from a one-dimensional, uniform subdivision
of elements.
"""
function child_to_parents_uniform(num_subdivisions::Int)
    _check_num_subdivsions(num_subdivisions)

    return RelationExplicit{Hierarchical.CP}(
        e -> _child_to_parents_uniform(e, num_subdivisions)
    )
end

function _child_to_parents_uniform(child_element_id, num_subdivisions)
    return (div(child_element_id + num_subdivisions - 1, num_subdivisions),)
end

function _check_num_subdivsions(num_subdivisions)
    foreach(_check_num_subdivsions, num_subdivisions)

    return nothing
end

function _check_num_subdivsions(num_subdivisions::Int)
    if num_subdivisions < 1
        throw(
            ArgumentError(
                LazyString(
                    "Number of subdivions must be greater than 0. Got ", num_subdivisions
                ),
            ),
        )
    end

    return nothing
end

# CartesianGeometry

function parent_to_children_uniform(
    geometry::CartesianGeometry{manifold_dim, image_dim, 1},
    num_subdivisions::NTuple{manifold_dim, Int},
) where {manifold_dim, image_dim}
    _check_num_subdivsions(num_subdivisions)
    parent_factor_num_elements = get_factor_num_elements(geometry, 1)
    child_factor_num_elements = parent_factor_num_elements .* num_subdivisions
    child_lin_ids = LinearIndices(child_factor_num_elements)

    # The function computing the children from a parent element id.
    function method(parent_element_id)
        # This method is dispatched when there is only one patch, so we call first.
        parent_factor_element_ids = first(
            get_factor_element_ids(geometry, parent_element_id)
        )
        child_factor_element_ids = ntuple(manifold_dim) do k
            return _parent_to_children_uniform(
                parent_factor_element_ids[k], num_subdivisions[k]
            )
        end

        # `flatten` is needed here because map will preserve the matrix structure of
        # `product`, but we want the children as a list of ids.
        return Iterators.flatten(
            Iterators.map(
                e -> child_lin_ids[e...], Iterators.product(child_factor_element_ids...)
            ),
        )
    end

    return RelationExplicit{Hierarchical.PC}(method)
end

function child_to_parents_uniform(
    geometry::CartesianGeometry{manifold_dim, image_dim, 1},
    num_subdivisions::NTuple{manifold_dim, Int},
) where {manifold_dim, image_dim}
    _check_num_subdivsions(num_subdivisions)
    child_factor_num_elements = get_factor_num_elements(geometry, 1)
    parent_factor_num_elements = div.(child_factor_num_elements, num_subdivisions)
    parent_lin_ids = LinearIndices(parent_factor_num_elements)

    # The function computing the parent from a child element id.
    function method(child_element_id)
        # This method is dispatched when there is only one patch, so we call first.
        child_factor_element_ids = first(get_factor_element_ids(geometry, child_element_id))
        parent_factor_element_ids = ntuple(manifold_dim) do k
            return _child_to_parents_uniform(
                child_factor_element_ids[k], num_subdivisions[k]
            )
        end

        # This method is dispatched when there is only one patch, so we call first on each
        # parent element.
        return (
            parent_lin_ids[Iterators.map(pe -> first(pe), parent_factor_element_ids)...],
        )
    end

    return RelationExplicit{Hierarchical.CP}(method)
end

############################################################################################
#                                         Scalings                                         #
############################################################################################

"""
    scaling_uniform(
        parent_geometry::AbstractGeometry{manifold_dim},
        child_geometry::AbstractGeometry{manifold_dim},
        num_subdivisions,
    ) where {manifold_dim}

Return a scaling from `parent_geometry` to `child_geometry`, resulting form a uniform
subdivison by `num_subdivisions`.

See [`Hierarchical.Scaling`](@ref).
"""
function scaling_uniform(
    parent_geometry::AbstractGeometry{manifold_dim},
    child_geometry::AbstractGeometry{manifold_dim},
    num_subdivisions,
) where {manifold_dim}
    relations = Relations(
        parent_to_children_uniform(parent_geometry, num_subdivisions),
        child_to_parents_uniform(child_geometry, num_subdivisions),
    )

    return Scaling(parent_geometry, child_geometry, relations)
end

"""
    scaling_uniform(parent_geometry::AbstractGeometry, num_subdivisions)

Return a scaling from `parent_geometry` to its child, resulting form a uniform subdivison by
`num_subdivisions`.

See also [`refinement_uniform`](@ref) and [`Hierarchical.Scaling`](@ref).
"""
function scaling_uniform(parent_geometry::AbstractGeometry, num_subdivisions)
    child_geometry = refinement_uniform(parent_geometry, num_subdivisions)
    relations = Relations(
        parent_to_children_uniform(parent_geometry, num_subdivisions),
        child_to_parents_uniform(child_geometry, num_subdivisions),
    )

    return Scaling(parent_geometry, child_geometry, relations)
end
