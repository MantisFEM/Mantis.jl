import Base: fill!, copyto!

############################################################################################
#                                          Buffer                                          #
############################################################################################

const CanonicalValues{T} = Array{T, 3}

struct CanonicalBuffer{T <: Number} <: Caching.AbstractBuffer
    values::CanonicalValues{T}
    status::Caching.Flag
    function CanonicalBuffer(values::CanonicalValues{T}) where {T}
        return new{T}(values, Caching.Flag(false))
    end
end

function Caching.preallocate(
    space::AbstractCanonicalSpace, points::Points.AbstractPoints{manifold_dim, T}
) where {manifold_dim, T}
    num_points = length(points)
    p = get_polynomial_degree(space)

    return CanonicalBuffer(CanonicalValues{T}(undef, (num_points, p + 1, 1)))
end

Caching.peek(buff::CanonicalBuffer) = buff.values

function fill!(
    buff::CanonicalBuffer,
    space::AbstractCanonicalSpace,
    element_id::Int,
    points::Points.AbstractPoints,
    der_key::Int,
)
    evaluate!(buff(), space, element_id, points, der_key)

    return buff
end

function Caching.clear!(buff::CanonicalBuffer{T}) where {T}
    fill!(buff(), zero(T))
    Caching.setfilled!(buff, false)

    return buff
end

function copyto!(dest_buff::CanonicalBuffer, src_buff::CanonicalBuffer)
    copyto!(dest_buff(), src_buff())

    return dest_buff
end

############################################################################################
#                                AbstractCanonicalSpace API                                #
############################################################################################

function evaluate!(
    values::CanonicalValues,
    space::AbstractCanonicalSpace,
    element_id::Int,
    points::Points.AbstractPoints,
    der_key::Int,
)
    space_eval = space_evaluate(space, element_id, der_key)
    # Loop over number of components, basis functions and points
    for c in axes(values, 3), basis in axes(values, 2), (p, point) in enumerate(points)
        @inbounds values[p, basis, c] = space_eval(point, basis, c)
    end

    return values
end

function space_evaluate(space::AbstractCanonicalSpace, element_id::Int, der_order::Int)
    throw(MethodError(space_evaluate, (space, element_id, der_order)))
end

include("Bernstein.jl")
