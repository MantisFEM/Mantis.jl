module Analysis

using ..Points
using ..Forms
using ..Quadrature

export L2_norm, Linf_norm, L2_norm_per_element, Linf_norm_per_element

include("ErrorComputations.jl")

end
