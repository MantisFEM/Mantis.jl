
"""
    MarkingStrategy

A strategy for marking objects to be updated from a hierarhical construction, for example
refinement or coarsening. The objects can be elements from the geometry, or functions from a
basis, for instance.
"""
abstract type MarkingStrategy end

"""
    (M::MarkingStrategy)(object_id::Int)

Return all the objects that are marked due to `object_id` being selected. Can be empty if no
objects are marked.
"""
function (marking::MarkingStrategy)(object_id::Int)
    return throw(MethodError(marking, (object_id,)))
end

"""
    (M::MarkingStrategy)(object_ids::AbstractVector)

Return all the objects that are marked due to `object_ids` being selected. Can be empty if
no objects are marked.
"""
function (marking::MarkingStrategy)(object_ids::AbstractVector)
    return mapreduce(marking, union, object_ids; init=Int[])
end

struct IdentityMarking <: MarkingStrategy end

(marking::IdentityMarking)(object_id::Int) = [object_id]

struct MaximumMarking{E <: AbstractVector{<:Real}, C <: Real} <: MarkingStrategy
    errors::E
    cutoff::C
    function MaximumMarking(errors::E, cutoff::Real) where {E}
        if any(<(0.0), errors)
            return throw(
                ArgumentError(LazyString("The errors should be non-negative. Got ", errors))
            )
        end

        if cutoff < 0.0 || cutoff > 1.0
            return throw(
                ArgumentError(
                    LazyString("The cutoff value should be between 0 and 1. Got ", cutoff)
                ),
            )
        end

        max_error = maximum(errors)
        # We will always do this computation, so we change the interpretation of cutoff.
        # This way the user can still provide the cutoff as a ratio, which is nicer.
        cutoff = max_error * cutoff

        return new{E, typeof(cutoff)}(errors, cutoff)
    end
end

function (marking::MaximumMarking)(object_id::Int)
    object_error = marking.errors[object_id]
    # the error is greater or equal than cutoff % of the maximum error
    if object_error >= marking.cutoff
        return [object_id]
    end

    # the error is smaller, so no element is marked due to `object_id`
    return Int[]
end

struct SupportMarking{
    S <: AbstractFunctionSpace, E <: AbstractVector{<:Real}, M <: MarkingStrategy
} <: MarkingStrategy
    space::S
    errors::E
    element_marking::M
    function SupportMarking(space::S, errors::E, element_marking::M) where {S, E, M}
        return new{S, E, M}(space, errors, element_marking)
    end
end

function (marking::SupportMarking)(object_id::Int)
    # We first compute what elements are marked from the single-element marking strategy.
    marked_elements = marking.element_marking(object_id)
    # Then we use those marked elements to determine what basis functions are supported on
    # those elements, and extend the marking with their supports.
    marked_elements = mapreduce(
        # For element `e`, we get the support of all basis functions `b` supported on `e`.
        e -> mapreduce(
            b -> get_support(marking.space, b),
            union,
            get_basis_indices(marking.space, e),
        ),
        union,
        marked_elements;
        init=Int[],
    )

    return marked_elements
end

"""
    DorflerMarking(errors::AbstractVector{<:Real}, cutoff::Real)

Dörfler marking. Marks a minimal set of objects `M` such that `sum(errors[M] .^ 2) >= cutoff
* sum(errors .^ 2)`, where `cutoff` is between 0 and 1.
Objects with the largest errors are marked first.
"""
struct DorflerMarking{E <: AbstractVector{<:Real}, C <: Real} <: MarkingStrategy
    errors::E
    cutoff::C
    function DorflerMarking(errors::E, cutoff::Real) where {E}
        if any(<(0.0), errors)
            return throw(
                ArgumentError(LazyString("The errors should be non-negative. Got ", errors))
            )
        end

        if cutoff < 0.0 || cutoff > 1.0
            return throw(
                ArgumentError(
                    LazyString("The cutoff value should be between 0 and 1. Got ", cutoff)
                ),
            )
        end

        # As in `MaximumMarking`, we change the interpretation of cutoff to an error
        # threshold, so that marking a single object only requires a comparison. The
        # threshold is the smallest error needed to reach `cutoff` of the total squared
        # error when accumulating errors from largest to smallest.
        target = cutoff * sum(abs2, errors; init=0.0)
        # If the target is zero, no objects need to be marked.
        threshold = typemax(float(eltype(errors)))
        if target > 0.0
            accumulated = zero(target)
            for error in sort(errors; rev=true)
                threshold = float(error)
                accumulated += error^2
                if accumulated >= target
                    break
                end
            end
        end

        return new{E, typeof(threshold)}(errors, threshold)
    end
end

const DörflerMarking = DorflerMarking

function (marking::DorflerMarking)(object_id::Int)
    # the error is among the largest ones needed to reach the bulk criterion
    if marking.errors[object_id] >= marking.cutoff
        return [object_id]
    end

    # the error is smaller, so no element is marked due to `object_id`
    return Int[]
end

"""
    QBoxMarking(space, qbox_size::NTuple{manifold_dim, Int}, element_marking::MarkingStrategy)

Marks all elements in the same qbox as the elements marked by `element_marking`. A qbox is a
block of `qbox_size[d]` consecutive elements in each dimension `d` of the parametric
geometry of `space`, which must be a `CartesianGeometry` or a `HierarchicalGeometry` of
`CartesianGeometry` levels. The number of elements in each dimension of each patch (and
level) must be a multiple of the corresponding `qbox_size`.

For a `HierarchicalGeometry`, the qbox of an element is computed on its level, and the
marked elements are given in hierarchical indexing. The hierarchy is assumed to be the
result of refining whole qboxes, so all elements in the qbox of an active element must be
active on the same level. An `ArgumentError` is thrown otherwise.
"""
struct QBoxMarking{
    manifold_dim,
    G <: Union{
        Geometry.CartesianGeometry{manifold_dim},
        Geometry.HierarchicalGeometry{manifold_dim},
    },
    M <: MarkingStrategy,
} <: MarkingStrategy
    geometry::G
    qbox_size::NTuple{manifold_dim, Int}
    element_marking::M
    function QBoxMarking(
        space, qbox_size::NTuple{manifold_dim, Int}, element_marking::M
    ) where {manifold_dim, M}
        geometry = get_parametric_geometry(space)
        if Geometry.get_manifold_dim(geometry) != manifold_dim
            return throw(
                ArgumentError(
                    LazyString(
                        "The qbox size should have one entry per dimension (",
                        Geometry.get_manifold_dim(geometry),
                        "). Got ",
                        qbox_size,
                    ),
                ),
            )
        end

        if any(<(1), qbox_size)
            return throw(
                ArgumentError(
                    LazyString("The qbox size should be positive. Got ", qbox_size)
                ),
            )
        end

        check_qbox_size(geometry, qbox_size)

        return new{manifold_dim, typeof(geometry), M}(geometry, qbox_size, element_marking)
    end
end

"""
    check_qbox_size(geometry, qbox_size::NTuple{manifold_dim, Int})

Throw an `ArgumentError` if `geometry` is not supported by `QBoxMarking`, or if its number
of elements is not a multiple of `qbox_size` in each dimension.
"""
function check_qbox_size(geometry, qbox_size)
    return throw(
        ArgumentError(
            LazyString(
                "The parametric geometry should be a CartesianGeometry or a ",
                "HierarchicalGeometry. Got ",
                typeof(geometry),
            ),
        ),
    )
end

function check_qbox_size(geometry::Geometry.CartesianGeometry, qbox_size)
    for patch_id in 1:Geometry.get_num_patches(geometry)
        num_elements = Geometry.get_factor_num_elements(geometry, patch_id)
        if any(!iszero, num_elements .% qbox_size)
            return throw(
                ArgumentError(
                    LazyString(
                        "The number of elements in each dimension should be a ",
                        "multiple of the qbox size. Got ",
                        num_elements,
                        " elements in patch ",
                        patch_id,
                        " and qbox size ",
                        qbox_size,
                    ),
                ),
            )
        end
    end

    return nothing
end

function check_qbox_size(geometry::Geometry.HierarchicalGeometry, qbox_size)
    # Every level should be a supported geometry, so we check them individually.
    for level in 1:Geometry.get_num_levels(geometry)
        level_geometry = Geometry.get_level_geometry(geometry, level)
        check_qbox_size(level_geometry, qbox_size)
    end

    return nothing
end

# element index and qbox size
qbox_id(i, s) = div(i - 1, s) + 1
# qbox index and qbox size
qbox_range(i, s) = ((i - 1) * s + 1):(i * s)

"""
    get_qbox_elements(marking::QBoxMarking, element_id::Int)

Return all the elements in the same qbox as `element_id`.
"""
function get_qbox_elements(marking::QBoxMarking, element_id::Int)
    return get_qbox_elements(marking.geometry, marking.qbox_size, element_id)
end

function get_qbox_elements(geometry::Geometry.CartesianGeometry, qbox_size, element_id::Int)
    factor_ids, patch_id = Geometry.get_factor_element_ids(geometry, element_id)
    q_range = map(factor_ids, qbox_size) do factor_id, size
        return qbox_range(qbox_id(factor_id, size), size)
    end

    local_ids = Geometry.get_lin_num_elements(geometry, patch_id)[q_range...]

    return [Geometry.get_global_element_id(geometry, patch_id, id) for id in vec(local_ids)]
end

function get_qbox_elements(
    geometry::Geometry.HierarchicalGeometry, qbox_size, element_id::Int
)
    # The qbox is computed on the level of the element, in level indexing.
    level, level_id = Geometry.convert_to_level_and_level_id(geometry, element_id)
    level_geometry = Geometry.get_level_geometry(geometry, level)
    qbox_level_ids = get_qbox_elements(level_geometry, qbox_size, level_id)
    # Then we convert back to hierarchical indexing. Since only whole qboxes are refined,
    # all the elements in the qbox have to be active on the same level.
    hierarchy = Geometry.get_hierarchy(geometry)
    level_set = Hierarchical.get_level_set(hierarchy, level)
    return map(qbox_level_ids) do qbox_level_id
        if !(qbox_level_id in level_set)
            return throw(
                ArgumentError(
                    LazyString(
                        "The element ",
                        qbox_level_id,
                        " of level ",
                        level,
                        " is in the qbox of element ",
                        element_id,
                        ", but is not active. Only hierarchies obtained by refining ",
                        "whole qboxes are supported.",
                    ),
                ),
            )
        end

        return Hierarchical.convert_to_hier_id(hierarchy, level, qbox_level_id)
    end
end

function (marking::QBoxMarking)(object_id::Int)
    # We first compute what elements are marked from the single-element marking strategy.
    marked_elements = marking.element_marking(object_id)
    # Then we extend the marking with all elements in the qbox of each marked element.
    return mapreduce(e -> get_qbox_elements(marking, e), union, marked_elements; init=Int[])
end
