import Base: fill!, copyto!

############################################################################################
#                                          Buffer                                          #
############################################################################################

const Values{T} = Matrix{T}

struct GeometryBuffer{T <: Number} <: Caching.AbstractBuffer
    values::Values{T}
    status::Caching.Flag
    function GeometryBuffer(values::Values{T}) where {T}
        new{T}(values, Caching.Flag(false))
    end
end

function Caching.preallocate(
    ::AbstractGeometry{manifold_dim, image_dim},
    points::Points.AbstractPoints{manifold_dim, T},
) where {manifold_dim, image_dim, T}
    num_points = length(points)

    return GeometryBuffer(Values{T}(undef, (num_points, image_dim)))
end

Caching.peek(buff::GeometryBuffer) = buff.values

function fill!(
    buff::GeometryBuffer,
    geometry::AbstractGeometry{manifold_dim},
    element_id::Int,
    points::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    evaluate!(buff(), geometry, element_id, points)

    return buff
end

function Caching.clear!(buff::GeometryBuffer{T}) where {T}
    fill!(buff(), zero(T))

    return buff
end

function copyto!(dest_buff::GeometryBuffer, src_buff::GeometryBuffer)
    copyto!(dest_buff(), src_buff())

    return dest_buff
end

############################################################################################
#                                       Geometry API                                       #
############################################################################################

function evaluate!(
    values::Values{T},
    geometry::AbstractGeometry{manifold_dim},
    element_id::Int,
    points::Points.AbstractPoints{manifold_dim},
) where {T, manifold_dim}
    point_eval = point_evaluate(geometry, element_id)
    for (p, point) in enumerate(points)
        @inbounds values[p, :] .= point_eval(point)
    end

    return values
end

############################################################################################
#                                         include                                          #
############################################################################################

include("CartesianGeometry.jl")
