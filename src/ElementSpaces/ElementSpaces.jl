"""
This (sub-)module provides a collection of polynomial and non-polynomial bases on the canonical element.

The exported names are:
"""
module ElementSpaces




"""
    AbstractElementSpace

Supertype for all element-local bases.
"""
abstract type AbstractElementSpace end


# Listed alphabetically
include("BernsteinPolynomials.jl")
include("LagrangePolynomials.jl")
include("ECTSpaces.jl")


# Has to be below the include statements to ensure that all evaluate 
# methods are visible to it!
@doc raw"""
    (elem_loc_basis::AbstractElementSpace)(xi::Vector{Float64}, args...)::Array{Float64}

Call the `evaluate`-method for the given `elem_loc_basis`.

Wrapper for all `evaluate`-methods so that all `AbstractElementSpace` can 
be called by calling the struct instead of explicitly calling the 
`evaluate`-method. Automatically throws a MethodError if the subtype 
does not have an evaluate function implemented. Some of these methods 
may have additional arguments, such as the additional derivatives to 
evaluate, so this version should allow that as well.

See also 
- [`evaluate(elem_loc_basis::AbstractLagrangePolynomials, ξ::Vector{Float64})`](@ref),
- [`evaluate(elem_loc_basis::Bernstein, xi::Vector{Float64}, nderivatives::Int64)`](@ref),
- [`evaluate(elem_loc_basis::Bernstein, xi::Vector{Float64})`](@ref),
- [`evaluate(elem_loc_basis::Bernstein, xi::Float64)`](@ref),
- [`evaluate(elem_loc_basis::Bernstein, xi::Float64, nderivatives::Int64)`](@ref).
"""
function (elem_loc_basis::AbstractElementSpace)(xi::Vector{Float64}, args...)::Array{Float64}
    return evaluate(elem_loc_basis, xi, args...)
end



end