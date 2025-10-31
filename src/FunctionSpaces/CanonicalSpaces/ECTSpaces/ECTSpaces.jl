
"""
    evaluate(ect_space::AbstractECTSpaces, ξ::Points.AbstractPoints{1}, nderivatives::Int=0)

Compute derivatives up to order `nderivatives` for all basis functions of degree `p` at `ξ`
for ``\\xi \\in [0.0, 1.0]``.

# Arguments
- `ect_space::AbstractECTSpaces`: ECT section space.
- `ξ::Points.AbstractPoints{1}`: vector of evaluation points ``\\in [0.0, 1.0]``.
- `nderivatives::Int`: maximum order of derivatives to be computed (nderivatives
    ``\\leq p``). Defaults to `0`, i.e., only the values are computed.
"""
Memoization.@memoize function evaluate(
    ect_space::AbstractECTSpaces, ξ::Points.AbstractPoints{1}, nderivatives::Int=0
)
    neval = Points.get_num_points(ξ)
    # allocate space for derivatives
    # - ders[j+1][1] contains the matrix of evaluations of the j-th derivative
    ders = Vector{Vector{Matrix{Float64}}}(undef, nderivatives + 1)
    for j in 0:nderivatives
        ders[j + 1] = Vector{Matrix{Float64}}(undef, 1)
        ders[j + 1][1] = zeros(Float64, neval, ect_space.p + 1)
    end
    # loop over the evaluation points and evaluate all derivatives at each point
    for i = 1:neval
        tmp = _evaluate(ect_space, ξ[i][1], nderivatives)
        for j = 0:nderivatives
            ders[j+1][1][i,:] .= tmp[1,:,j+1]
        end
    end
    return ders
end

include("GeneralizedTrigonometric.jl")
include("GeneralizedExponential.jl")
include("Tchebycheff.jl")
