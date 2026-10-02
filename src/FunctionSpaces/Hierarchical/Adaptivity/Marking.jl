
"""
    MarkingStrategy{S}

A strategy for marking objects to be updated from a hierarhical construction, for example
refinement or coarsening. The objects can be elements from the geometry, or functions from a
basis, for instance.

# Type Parameters
- `S`: Type of the set of objects being marked.
"""
abstract type MarkingStrategy{S, V, C} end

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
    return throw(MethodError(marking, (object_ids,)))
end

struct IdentityMarking{S} <: MarkingStrategy{S}
    set::S
end

# We split the methods (instead of using `Union`) to avoid an ambiguous dispatch witht the
# abstract method.
(marking::IdentityMarking)(object_id::Int) = object_id
(marking::IdentityMarking)(object_ids::AbstractVector) = object_ids

# TODO: Implement these marking strategies

struct MaximumMarking{S, E, C <: Real} <: MarkingStrategy{S}
    set::S
    errors::E
    cutoff::C
    function MaximumMarking(
        set::S, errors::E, cutoff::Real
    ) where {S, E <: AbstractVector{<:Real}}
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
        # we will always do this computation, so we change the interpretation of cutoff.
        # that way the user can still provide the cutoff as a ratio, which is nicer.
        cutoff = max_error * cutoff

        return new{S, E, typeof(cutoff)}(set, errors, max_error, cutoff)
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

struct SupportMarking{S} <: MarkingStrategy{S}

# struct DorflerMarking <: MarkingStrategy
#     cutoff::Float64
# end
#
# const DörflerMarking = DorflerMarking
#
# struct QBoxMarking <: MarkingStrategy end
