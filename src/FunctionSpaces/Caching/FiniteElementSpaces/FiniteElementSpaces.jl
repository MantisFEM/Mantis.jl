import Base: fill!, copyto!

############################################################################################
#                                          Buffer                                          #
############################################################################################

const FEValues{T} = Array{T, 3}

struct FEBuffer{T <: Number, CC} <: Caching.AbstractBuffer
    values::FEValues{T}
    canonical_cache::CC
    status::Caching.Flag
    function FEBuffer(values::FEValues{T}, canonical_cache::CC) where {T, CC}
        return new{T, CC}(values, canonical_cache, Caching.Flag(false))
    end
end

get_canonical_cache(buff::FEBuffer) = buff.canonical_cache

function Caching.preallocate(
    space::AbstractFESpace, points::Points.AbstractPoints{manifold_dim, T}
) where {manifold_dim, T}
    num_points = length(points)
    num_basis = get_max_local_dim(space)
    num_components = get_num_components(space)
    values = FEValues{T}(undef, (num_points, num_basis, num_components))
    canonical_cache = Caching.Cache(get_canonical_space(space), points)

    return FEBuffer(values, canonical_cache)
end

Caching.peek(buff::FEBuffer) = buff.values

function fill!(
    buff::FEBuffer,
    space::AbstractFESpace,
    element_id::Int,
    points::Points.AbstractPoints,
    der_key::Int,
)
    canonical_cache = get_canonical_cache(buff)
    Caching.update!(canonical_cache, element_id, points, der_key)
    evaluate!(buff(), space, element_id, canonical_cache())

    return buff
end

function Caching.clear!(buff::FEBuffer{T}) where {T}
    fill!(buff(), zero(T))
    Caching.setfilled!(buff, false)

    return buff
end

function copyto!(dest_buff::FEBuffer, src_buff::FEBuffer)
    copyto!(dest_buff(), src_buff())

    return dest_buff
end

############################################################################################
#                                         evaluate                                         #
############################################################################################

function evaluate!(
    values::FEValues,
    space::AbstractFESpace,
    element_id::Int,
    canonical_values::CanonicalValues,
)
    for c in axes(values, 3)
        E, J = get_extraction(space, element_id, c)
        @views LinearAlgebra.mul!(values[:, J, c], canonical_values[:, :, c], E)
    end

    return values
end

############################################################################################
#                                         include                                          #
############################################################################################

include("UnivariateSplines/UnivariateSplines.jl")
