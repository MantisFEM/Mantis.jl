"""
    module Caching

Contains structure and method definitions related with caching `Mantis` objects. Caching
refers to the pre-allocation of buffers which get filled in-place for various
`evaluate`-type methods. Examples of objects include `FunctionSpaces.AbstractFESpace`, or
`Geometry.AbstractGeometry`. As these are only defined in modules loaded after `Caching`,
the current module mainly contains place-holders for concrete methods, with general
doc-strings, and the main structure [`Cache`](@ref).
"""
module Caching

import Base: copy, deepcopy, fill!, copyto!, map, map!

############################################################################################
#                                         Exports                                          #
############################################################################################

# Internal exports.
export Cache, get_object, get_buffer, peek, preallocate, fill!, clear!
# Public exports.
include("CachingExports.jl")

############################################################################################
#                                      Abstract types                                      #
############################################################################################

"""
    AbstractCache{O, B}

Abstract type for all cache structure combining an object `::O` and its buffer `::B`.
"""
abstract type AbstractCache{O, B} end

"""
    AbstractBuffer

Abstract type for all buffer structures.
"""
abstract type AbstractBuffer end

"""
    AbstractStatus

Abstract type for all buffer status indicators. See also [`AbstractBuffer`](@ref).
"""
abstract type AbstractStatus end

############################################################################################
#                                          Cache                                           #
############################################################################################

"""
    Cache{O, B}

Contains an `object::O`, and a pre-allocated `buffer`. Evaluations on `object` can be
written in-place, overwriting the contents of `buffer`.

# Fields
- `object::O`: The variable used for performing evaluations.
- `buffer::B`: The pre-allocated buffer to store the outputs of evaluating on `object`.
"""
struct Cache{O, B} <: AbstractCache{O, B}
    object::O
    buffer::B

    function Cache(object::O, buffer::B) where {O, B <: AbstractBuffer}
        return new{O, B}(object, buffer)
    end
end

"""
    Cache(object)

Initialise a `Cache` for an `object` that fully embeds pre-allocation information. An
example is `Forms.Integral`, which contains a global quadrature rule.
"""
Cache(object) = Cache(object, preallocate(object))

"""
    Cache(object, args...)

Initialise a `Cache` for an `object`, using information from `args`.
"""
Cache(object, args...) = Cache(object, preallocate(object, args...))

############################################################################################
#                                    AbstractCache API                                     #
############################################################################################

"""
    extract(cache::AbstractCache)

Return the object and buffer from `cache`.
"""
extract(cache::AbstractCache) = get_object(cache), get_buffer(cache)

"""
    get_object(cache::Cache)

Returns the `object` from `cache`.
"""
get_object(cache::AbstractCache) = cache.object

"""
    get_buffer(cache::Cache)

Return the `buffer` from `cache`.
"""
get_buffer(cache::AbstractCache) = cache.buffer

"""
    peek(cache::AbstractCache)

Return contents of the `buffer` in `cache`.
"""
peek(cache::AbstractCache) = peek(get_buffer(cache))

(cache::AbstractCache)() = peek(cache)

"""
    update!(cache::AbstractCache, args...)

Similar to `fill!(cache)`, but immediatelly returns `cache` if `isfilled` is true.
Else, calls `fill!(cache, args...)` followed by `setfilled!(cache, true)`.
"""
function update!(cache::AbstractCache, args...)
    obj, buff = extract(cache)
    update!(buff, obj, args...)

    return cache
end

function fill!(cache::AbstractCache, args...)
    obj, buff = extract(cache)
    fill!(buff, obj, args...)

    return cache
end

(cache::AbstractCache)(args...) = fill!(cache, args...)

"""
    clear!(cache::AbstractCache)

Overwrite the contents of `cache` with zeros.
"""
function clear!(cache::AbstractCache)
    cache |> get_buffer |> clear!

    return cache
end

"""
    status(cache::AbstractCache)

Return the status of `cache`, of type `AbstractStatus`.
"""
status(cache::AbstractCache) = status(get_buffer(cache))

"""
    isfilled(cache::AbstractCache)

Return `true` if the cache is filled, and `false` otherwise.
"""
isfilled(cache::AbstractCache) = isfilled(get_buffer(cache))

"""
    setfilled!(cache::AbstractCache, args...)

Update the status of `cache` with the provided `args...`.
"""
function setfilled!(cache::AbstractCache, args...)
    setfilled!(get_buffer(cache), args...)

    return cache
end

############################################################################################
#                                        Buffer API                                        #
############################################################################################

# pre-allocation

"""
    preallocate(object)

Pre-allocate a buffer for `object`.
"""
function preallocate(object)
    throw(MethodError(preallocate, (object,)))
end

"""
    preallocate(object, args...)

Pre-allocate a buffer for `object`, using information from `args`.
"""
function preallocate(object, args...)
    throw(MethodError(preallocate, (object, args...)))
end

# information operations

"""
    peek(buffer::AbstractBuffer)

Return contents of `buffer`.

!!! note
    Each concrete type of `AbstractBuffer` determines what the contents are; they need not
    match every field of `buffer`.
"""
function peek(buffer::AbstractBuffer)
    throw(MethodError(peek, (buffer,)))
end

(buff::AbstractBuffer)() = peek(buff)

function copy(buff::AbstractBuffer)
    throw(MethodError(copy, (buff,)))
end

function deepcopy(buff::AbstractBuffer)
    throw(MethodError(deepcopy, (buff,)))
end

# overwriting operations

"""
    update!(buff::AbstractBuffer, args...)

Similar to `fill!(buff)`, but immediatelly returns `buff` if `isfilled` is true.
Else, calls `fill!(buff, args...)` followed by `setfilled!(buff, true)`.
"""
function update!(buff::AbstractBuffer, args...)
    if !isfilled(buff)
        fill!(buff, args...)
        setfilled!(buff, true)
    end

    return buff
end

function fill!(buffer::AbstractBuffer, args...)
    throw(MethodError(fill!, (buffer, args...)))
end

(buffer::AbstractBuffer)(args...) = fill!(buffer, args...)

"""
    clear!(buffer::AbstractBuffer)

Overwrite the contents of `buffer` with zeros.
"""
function clear!(buffer::AbstractBuffer)
    throw(MethodError(clear!, (buffer,)))
end

function copyto!(dest_buffer::AbstractBuffer, src_buffer::AbstractBuffer)
    throw(MethodError(copyto!, (dest_buffer, src_buffer)))
end

"""
    status(buff::AbstractBuffer)

Return the status of `buff`, of type `AbstractStatus`.
"""
status(buff::AbstractBuffer) = buff.status

"""
    isfilled(buff::AbstractBuffer)

Return `true` if the buff is filled, and `false` otherwise.
"""
isfilled(buff::AbstractBuffer) = status(buff)()

"""
    setfilled!(buff::AbstractBuffer, condition::Bool)

Update the status of `buff` with the provided `args...`.
"""
setfilled!(buff::AbstractBuffer, args...) = set!(status(buff), args...)

############################################################################################
#                                           Flag                                           #
############################################################################################

"""
    Flag <: AbstractStatus

Boolean status indicator.
"""
mutable struct Flag <: AbstractStatus
    condition::Bool
end

############################################################################################
#                                    AbstractStatus API                                    #
############################################################################################

(status::AbstractStatus)() = status.condition

function set!(status::AbstractStatus, condition::Bool)
    setfield!(status, :condition, condition)

    return status
end

############################################################################################
#                                     CompositeBuffer                                      #
############################################################################################

"""
    CompositeBuffer{B <: Tuple} <: AbstractBuffer

A composite buffer consisting of a fixed-length, ordered collection of
[`AbstractBuffer`](@ref) sub-buffers.  Each sub-buffer is accessed via
[`get_buffers`](@ref), and operations such as [`peek`](@ref), [`fill!`](@ref), and
[`clear!`](@ref) are delegated to every sub-buffer in order.

# Type parameters
- `B <: Tuple`: Encodes the concrete type of every sub-buffer.

# Fields
- `buffers::B`: The tuple of sub-buffers.
"""
struct CompositeBuffer{B <: Tuple} <: AbstractBuffer
    buffers::B

    function CompositeBuffer(buffers::B) where {B <: Tuple}
        all(b -> b isa AbstractBuffer, buffers) || throw(
            ArgumentError(
                "All elements of `buffers` must be subtypes of `AbstractBuffer`."
            ),
        )
        return new{B}(buffers)
    end
end

"""
    CompositeBuffer(buffers::AbstractBuffer...)

Construct a [`CompositeBuffer`](@ref) from a variable number of
[`AbstractBuffer`](@ref) arguments.
"""
CompositeBuffer(buffers::AbstractBuffer...) = CompositeBuffer(buffers)

"""
    get_buffers(buff::CompositeBuffer)

Return the tuple of sub-buffers stored in `buff`.
"""
get_buffers(buff::CompositeBuffer) = buff.buffers

############################################################################################
#                                  CompositeBuffer API                                     #
############################################################################################

peek(buff::CompositeBuffer) = map(peek, buff)

map(f, buff::CompositeBuffer) = map(f, get_buffers(buff))

function map!(f, buff::CompositeBuffer)
    foreach(sub_buff -> f(sub_buff), get_buffers(buff))

    return buff
end

function map!(f, buff::CompositeBuffer, i)
    f(get_buffers(buff)[i])

    return buff
end

copy(buff::CompositeBuffer) = CompositeBuffer(map(copy, buff))

deepcopy(buff::CompositeBuffer) = CompositeBuffer(map(deepcopy, buff))

function copyto!(dest::CompositeBuffer{B}, src::CompositeBuffer{B}) where {B}
    foreach(copyto!, get_buffers(dest), get_buffers(src))

    return dest
end

isfilled(buff::CompositeBuffer) = all(b -> isfilled(b), get_buffers(buff))

############################################################################################
#                                          Macros                                          #
############################################################################################

"""
    cached(expr)

Given a function call, `expr`, on an underlying `object`, assumed to be the first argument,
generates a [`Cache`](@ref) for `object`, modifies `expr` to an in-place function, by
appending `!` to the function's signature, calls the modified function on the cache of
`object`, and returns the cache.

# Arguments
- `expr`: The function call to convert to an in-place-modifying version.

# Returns
- `Cache`: The instantiated cache object, overwritten with the output of `expr`.

# Examples
The following two code blocks are equivalent:
```julia
cache = @cached evaluate(obj, element, points)
```
and
```julia
cache = Cache(obj, points)
evaluate!(cache, element, points)
```
"""
macro cached(expr)
    Meta.isexpr(expr, :call) || throw(ArgumentError("@cached expects a function call."))
    # Arguments of the function call, without the signature.
    args = map(esc, expr.args[2:end])
    expr! = esc(_append_bang(expr.args[1]))
    quote
        #=
        Instantiates a `Cache` with the object, assumed to be the first argument, and the
        remaining arguments after the second, since the second parameter is usually
        `element_id`.
        This is the case for our `evaluate` calls: `evaluate(obj, element_id, xi, ...)`.
        =#
        cache = Cache($(args[1]), $(args[3:end]...))
        output = $(expr!)(cache, $(args[2:end]...))
        output
    end
end

function _append_bang(expr)
    # `expr` is a function signature. For example, `evaluate`.
    if expr isa Symbol
        return Symbol(string(expr), "!")
    end

    # `expr` is a function signature after a module name. For example, `Geometry.evaluate`.
    if Meta.isexpr(expr, :.) &&
        expr.args[1] isa Symbol &&
        expr.args[2] isa QuoteNode &&
        expr.args[2].value isa Symbol
        mod = expr.args[1]
        sig = expr.args[2].value

        return Expr(:., mod, QuoteNode(Symbol(string(sig), "!")))
    end

    throw(
        ArgumentError(
            "Unsupported call for @cached. Only `foo(...)` or `Module.foo(...)` is supported",
        ),
    )
end

end
