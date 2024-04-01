"""
This (sub-)module provides a collection of polynomial bases.

The exported names are:
"""
module Polynomials




"""
    AbstractPolynomials

Supertype for all polynomials.
"""
abstract type AbstractPolynomials end


# Listed alphabetically
include("BernsteinPolynomial.jl")
include("LagrangePolynomials.jl")
include("ECTSpaces.jl")


# Has to be below the include statements to ensure that all evaluate 
# methods are visible to it!
@doc raw"""
    (polynomial::AbstractPolynomials)(xi::Vector{Float64}, args...)::Array{Float64}

Call the `evaluate`-method for the given `polynomial`.

Wrapper for all `evaluate`-methods so that all `AbstractPolynomials` can 
be called by calling the struct instead of explicitly calling the 
`evaluate`-method. Automatically throws a MethodError if the subtype 
does not have an evaluate function implemented. Some of these methods 
may have additional arguments, such as the additional derivatives to 
evaluate, so this version should allow that as well.

See also 
- [`evaluate(polynomial::AbstractLagrangePolynomials, ξ::Vector{Float64})`](@ref),
- [`evaluate(polynomial::Bernstein, xi::Vector{Float64}, nderivatives::Int64)`](@ref),
- [`evaluate(polynomial::Bernstein, xi::Vector{Float64})`](@ref),
- [`evaluate(polynomial::Bernstein, xi::Float64)`](@ref),
- [`evaluate(polynomial::Bernstein, xi::Float64, nderivatives::Int64)`](@ref).
"""
function (polynomial::AbstractPolynomials)(xi::Vector{Float64}, args...)::Array{Float64}
    return evaluate(polynomial, xi, args...)
end



end