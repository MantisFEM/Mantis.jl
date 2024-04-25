"""
    AbstractCanonicalSpace

Supertype for all element-local bases.
"""
abstract type AbstractCanonicalSpace <: AbstractFunctionSpace end

# Listed alphabetically
include("BernsteinPolynomials.jl")
include("LagrangePolynomials.jl")
include("ECTSpaces.jl")


# # Has to be below the include statements to ensure that all evaluate 
# # methods are visible to it!
# @doc raw"""
#     (canonical_space::AbstractCanonicalSpace)(xi::Vector{Float64}, args...)::Array{Float64}

# Call the `evaluate`-method for the given `canonical_space`.

# Wrapper for all `evaluate`-methods so that all `AbstractCanonicalSpace` can 
# be called by calling the struct instead of explicitly calling the 
# `evaluate`-method. Automatically throws a MethodError if the subtype 
# does not have an evaluate function implemented. Some of these methods 
# may have additional arguments, such as the additional derivatives to 
# evaluate, so this version should allow that as well.

# See also 
# - [`evaluate(canonical_space::AbstractLagrangePolynomials, ξ::Vector{Float64})`](@ref),
# - [`evaluate(canonical_space::Bernstein, xi::Vector{Float64}, nderivatives::Int64)`](@ref),
# - [`evaluate(canonical_space::Bernstein, xi::Vector{Float64})`](@ref),
# - [`evaluate(canonical_space::Bernstein, xi::Float64)`](@ref),
# - [`evaluate(canonical_space::Bernstein, xi::Float64, nderivatives::Int64)`](@ref).
# """
function (canonical_space::T where {T <: AbstractCanonicalSpace})(xi::Vector{Float64}, args...)
    return evaluate(canonical_space, xi, args...)
end