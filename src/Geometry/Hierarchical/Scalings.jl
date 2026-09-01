
############################################################################################
#                                        Relations                                         #
############################################################################################

# 1D Generic

function parent_to_children_uniform(::AbstractGeometry{1}, num_subdivisions::Int)
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

function child_to_parents_uniform(num_subdivisions::Int)
    _check_num_subdivsions(num_subdivisions)

    return RelationExplicit{Hierarchical.CP}(
        e -> _child_to_parents_uniform(e, num_subdivisions)
    )
end

function _child_to_parents_uniform(child_element_id, num_subdivisions)
    return (div(child_element_id + num_subdivisions - 1, num_subdivisions),)
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

    function method(parent_element_id)
        parent_factor_element_ids = first(
            get_factor_element_ids(geometry, parent_element_id)
        )
        child_factor_element_ids = ntuple(manifold_dim) do k
            return _parent_to_children_uniform(
                parent_factor_element_ids[k], num_subdivisions[k]
            )
        end

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

    function method(child_element_id)
        child_factor_element_ids = first(get_factor_element_ids(geometry, child_element_id))
        parent_factor_element_ids = ntuple(manifold_dim) do k
            return _child_to_parents_uniform(
                child_factor_element_ids[k], num_subdivisions[k]
            )
        end

        return Iterators.flatten(
            Iterators.map(
                e -> parent_lin_ids[e...], Iterators.product(parent_factor_element_ids...)
            ),
        )
    end

    return RelationExplicit{Hierarchical.CP}(method)
end

function _check_num_subdivsions(num_subdivisions)
    if any(<(1), num_subdivisions)
        throw(
            ArgumentError(
                LazyString(
                    "Number of subdivions must be greater than 0. Got ", num_subdivisions
                ),
            ),
        )
    end
end

function scaling_uniform(parent_geometry, child_geometry, num_subdivisions)
    relations = Relations(
        parent_to_children_uniform(parent_geometry, num_subdivisions),
        child_to_parents_uniform(child_geometry, num_subdivisions),
    )

    return Scaling(parent_geometry, child_geometry, relations)
end

function scaling_uniform(parent_geometry, num_subdivisions)
    child_geometry = refinement_uniform(parent_geometry, num_subdivisions)
    relations = Relations(
        parent_to_children_uniform(parent_geometry, num_subdivisions),
        child_to_parents_uniform(child_geometry, num_subdivisions),
    )

    return Scaling(parent_geometry, child_geometry, relations)
end
