"""
    GeneralizedTrigonometric <: AbstractECTSpaces

Concrete type for Generalized Trignometric section space spanned by
``<1, x, ..., x^(p-2), cos(wx), sin(wx)>`` on ``[0,l]``; see [Schumaker2007](@cite).

# Fields
- `p::Int`: Degree of the space.
- `w::Float64`: Weight parameter for the space.
- `l::Float64`: Length of the interval. GTrig space is not scale-invariant.
- `t::Bool`: flag to indicate if critical length is exceeded.
- `m::Int`: number of terms from the infinite sum used to build the basis.
- `C::Matrix{Float64}`: representation matrix for the local basis.
"""
struct GeneralizedTrigonometric <: AbstractECTSpaces
    p::Int
    w::Float64
    l::Float64
    t::Bool
    m::Int
    C::Matrix{Float64}
    endpoint_tol::Float64

    function GeneralizedTrigonometric(p::Int, w::Float64=1.0, l::Float64=1.0, m::Int=10)
        t = abs(w) * l >= 3.0
        return GeneralizedTrigonometric(p, w, l, t, m)
    end

    function GeneralizedTrigonometric(p::Int, w::Float64, l::Float64, t::Bool, m::Int)
        endpoint_tol = 1e-12
        if p < 1
            throw(ArgumentError("Degree p must be a positive integer."))
        end
        new(p, w, l, t, m, gtrig_representation(p, w, l, t, m), endpoint_tol)
    end
end

function _evaluate(ect_space::GeneralizedTrigonometric, xi::Float64, nderivatives::Int)
    M = zeros(Float64, 1, ect_space.p+1, nderivatives+1)

    left = false
    right = false
    if xi < ect_space.endpoint_tol
        left = true
    elseif xi > 1.0 - ect_space.endpoint_tol
        right = true
    end

    # scale the point to lie in the interval [0, l]
    xi = ect_space.l * xi
    if ect_space.t
        for k = 0:nderivatives
            wxl = ect_space.w * xi
            E = [1.0; cumprod((1.0 ./ (1:ect_space.p-k)) * wxl)]
            if mod(k, 4) == 0
                E[ect_space.p-k] = cos(wxl);
                E[ect_space.p+1-k] = sin(wxl);
            elseif mod(k, 4) == 1
                E[ect_space.p-k] = -sin(wxl);
                E[ect_space.p+1-k] = cos(wxl);
            elseif mod(k, 4) == 2
                E[ect_space.p-k] = -cos(wxl);
                E[ect_space.p+1-k] = -sin(wxl);
            elseif mod(k, 4) == 3
                E[ect_space.p-k] = sin(wxl);
                E[ect_space.p+1-k] = -cos(wxl);
            end
            # rescale the derivative to map back from [0, l] -> [0, 1]
            M[1, :, k+1] = (ect_space.w^k) * (ect_space.C[:,k+1:end] * E) * (ect_space.l^k)
        end
    else
        for k = 0:nderivatives
            ww = [1; cumprod(repeat([-ect_space.w * ect_space.w], ect_space.m))]
            Ef = [1.0; cumprod((1.0 ./ (1:ect_space.p-k+2*ect_space.m)) * xi)]
            E = Ef[1:ect_space.p+1-k]
            E[ect_space.p-k, :] = Ef[ect_space.p-k:2:end, :]' * ww
            E[ect_space.p-k+1, :] = Ef[ect_space.p-k+1:2:end, :]' * ww
            # rescale the derivative to map back from [0, l] -> [0, 1]
            M[1, :, k+1] = ect_space.C[:,k+1:end] * E * (ect_space.l^k)
        end
    end

    if left
        M[1, :, :] = LinearAlgebra.triu(M[1, :, :])
    elseif right
        M[1, :, :] = LinearAlgebra.triu!(M[1, end:-1:1, :])[end:-1:1, :]
    end

    return M
end

"""
    evaluate(ect_space::GeneralizedTrigonometric, ξ::Vector{Float64})

Compute all basis function values at `ξ` in ``[0.0, 1.0]``.

# Arguments
- `ect_space::GeneralizedTrigonometric`:  Generalized Trigonometric section space.
- `xi::Vector{Float64}`: vector of evaluation points in ``[0.0, 1.0]``.

See also [`evaluate(ect_space::GeneralizedTrigonometric, xi::Vector{Float64}, nderivatives::Int64)`](@ref).
"""
function evaluate(ect_space::GeneralizedTrigonometric, xi::Vector{Float64})
    return evaluate(ect_space, xi, 0)
end

function evaluate(ect_space::GeneralizedTrigonometric, xi::Float64)
    return evaluate(ect_space, [xi], 0)
end

"""
    gtrig_representation(p::Int, w::Float64, t::Bool, m::Int)

Build representation matrix for Generalized Trignometric section space of degree `p`, weight
`w` and length `l`.

# Arguments
- `p::Int`: Degree of the space.
- `w::Float64`: Weight parameter for the space.
- `l::Float64`: Length of the interval. GTrig space is not scale-invariant.
- `t::Bool`: flag to indicate if critical length is exceeded.
- `m::Int`: number of terms from the infinite sum used to build the basis.

# Returns:
- `C::Matrix{Float64}`: representation matrix for the local basis.
"""

function gtrig_representation(p::Int, w::Float64, l::Float64, t::Bool, m::Int)
    I = Matrix(1.0LinearAlgebra.I, p + 1, p + 1)
    if t
        wl = w * l
        cwl = cos(wl)
        swl = sin(wl)
        M0 = I[:, :]
        M0[[p, p + 1], [p, p + 1]] .= 0.0
        M0[p, 1:4:end] .= 1
        M0[p, 3:4:end] .= -1
        M0[p + 1, 2:4:end] .= 1
        M0[p + 1, 4:4:end] .= -1
        M1 = Matrix(ToeplitzMatrices.Toeplitz([1; cumprod(wl ./ (1:p))], I[:, 1]))
        M1[p, 1:4:end] .= cwl
        M1[p, 2:4:end] .= -swl
        M1[p, 3:4:end] .= -cwl
        M1[p, 4:4:end] .= swl
        M1[p + 1, 1:4:end] .= swl
        M1[p + 1, 2:4:end] .= cwl
        M1[p + 1, 3:4:end] .= -swl
        M1[p + 1, 4:4:end] .= -cwl
    else
        M0 = I[:, :]
        ww = [1 cumprod(repeat([-w * w], 1, m); dims=2)]
        M = ToeplitzMatrices.Toeplitz([1; cumprod(l ./ (1:(p + 2 * m)))], I[:, 1])
        M1 = M[1:(p + 1), :]
        M1[p, :] = ww * M[p:2:end, :]
        M1[p + 1, :] = ww * M[(p + 1):2:end, :]
    end
    cs = zeros(Float64, p + 1)
    C = zeros(Float64, p + 1, p + 1)
    C[p + 1, :] = [M1[:, 1] M0[:, 1:p]]' \ [1.0; zeros(p)]
    for i in 2:p
        cs = cs + C[p + 1 - i + 2, :]
        cc = zeros(Float64, p + 1)
        cc[i] = -cs' * M1[:, i]
        C[p + 1 - i + 1, :] = [M1[:, 1:i] M0[:, 1:(p + 1 - i)]]' \ cc
    end
    C[1, :] = [M0[:, 1] M1[:, 1:p]]' \ [1.0; zeros(p)]

    return C
end

"""
    get_derivative_space(ect_space::GeneralizedTrigonometric)

Get the space of one degree lower than the input space. Assumes that the degree of the space
is at least 2.

# Arguments
- `ect_space::GeneralizedTrigonometric`: A generalized trigonometric space.

# Returns
- `::GeneralizedTrigonometric`: A generalized trigonometric space of one degree lower than the input space.
"""
function get_derivative_space(ect_space::GeneralizedTrigonometric)
    if ect_space.p < 2
        throw(ArgumentError("Degree of the space must be at least 2 to get derivative space."))
    end
    return GeneralizedTrigonometric(
        ect_space.p - 1, ect_space.w, ect_space.l, ect_space.m
    )
end

function get_canonical_space_on_subelements(
    space::GeneralizedTrigonometric; num_subdivisions::Int = 2, degree_delta::Int = 0
)   
    return GeneralizedTrigonometric(
        space.p + degree_delta, space.w, space.l / num_subdivisions, space.m
    )
end