"""
    GeneralizedExponential <: AbstractECTSpaces

Concrete type for Generalized Exponential section space spanned by
`<1, x, ..., x^(p-2), exp(wx), exp(-wx)>`, equivalently
`<1, x, ..., x^(p-2), cosh(wx), sinh(wx)>`, on `[0,l]`.

# Fields
- `p::Int`: Degree of the space.
- `w::Float64`: Weight parameter for the space.
- `l::Float64`: Length of the interval. GExp space is not scale-invariant.
- `t::Bool`: flag to indicate if critical length is exceeded.
- `m::Int`: number of terms from the infinite sum used to build the basis.
- `C::Matrix{Float64}`: representation matrix for the local basis.
"""
struct GeneralizedExponential <: AbstractECTSpaces
    p::Int
    w::Float64
    l::Float64
    t::Bool
    m::Int
    C::Matrix{Float64}
    endpoint_tol::Float64

    function GeneralizedExponential(p::Int, w::Float64=1.0, l::Float64=1.0, m::Int=10)
        t = abs(w) * l >= 3.0
        return GeneralizedExponential(p, w, l, t, m)
    end

    function GeneralizedExponential(p::Int, w::Float64, l::Float64, t::Bool, m::Int)
        endpoint_tol = 1e-12
        if p < 1
            throw(ArgumentError("Degree p must be a positive integer."))
        end
        new(p, w, l, t, m, gexp_representation(p, w, l, t, m), endpoint_tol)
     end
end

function _evaluate(ect_space::GeneralizedExponential, xi::Float64, nderivatives::Int)
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
        for r = 0:nderivatives
            k = min(r, ect_space.p-1)
            wxl = ect_space.w * xi
            E = [1.0; cumprod((1.0 ./ (1:ect_space.p-k)) * wxl)]
            E[ect_space.p-k, :] .= exp(wxl);
            E[ect_space.p+1-k, :] .= (-1)^r * exp(-wxl);
            # rescale the derivative to map back from [0, l] -> [0, 1]
            M[1, :, r+1] = (ect_space.w^r) * (ect_space.C[:,k+1:end] * E) * (ect_space.l^r)
        end
    else
        for r = 0:nderivatives
            k = min(r, ect_space.p-1)
            ww = [1; cumprod(repeat([ect_space.w * ect_space.w], ect_space.m))]
            Ef = [1.0; cumprod((1.0 ./ (1:ect_space.p-k+2*ect_space.m)) * xi)]
            E = Ef[1:ect_space.p+1-k]
            E[ect_space.p-k, :] = Ef[ect_space.p-k:2:end, :]' * ww
            E[ect_space.p-k+1, :] = Ef[ect_space.p-k+1:2:end, :]' * ww
            # rescale the derivative to map back from [0, l] -> [0, 1]
            M[1, :, r+1] = ect_space.C[:,k+1:end] * E * (ect_space.l^r)
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
    evaluate(ect_space::GeneralizedExponential, ξ::Vector{Float64})

Compute all basis function values at `ξ` in ``[0.0, 1.0]``.

# Arguments
- `ect_space::GeneralizedExponential`:  Generalized Exponential section space.
- `xi::Vector{Float64}`: vector of evaluation points ``in [0.0, 1.0]``.
"""
function evaluate(ect_space::GeneralizedExponential, xi::Vector{Float64})
    return evaluate(ect_space, xi, 0)
end

function evaluate(ect_space::GeneralizedExponential, xi::Float64)
    return evaluate(ect_space, [xi], 0)
end

"""
    gexp_representation(p::Int, w::Float64, t::Bool, m::Int)

Build representation matrix for Generalized Exponential section space of degree `p`, weight
`w`, and length `l`.

# Arguments
- `p::Int`: Degree of the space.
- `w::Float64`: Weight parameter for the space.
- `l::Float64`: Length of the interval. GExp space is not scale-invariant.
- `t::Bool`: flag to indicate if critical length is exceeded.
- `m::Int`: number of terms from the infinite sum used to build the basis.

# Returns:
- `C::Matrix{Float64}`: representation matrix for the local basis.
"""
function gexp_representation(p::Int, w::Float64, l::Float64, t::Bool, m::Int)
    I = Matrix(1.0LinearAlgebra.I, p + 1, p + 1)
    if t
        wl = w * l
        ewl = exp(wl)
        ewlm = exp(-wl)
        M0 = I[:, :]
        M0[p, :] .= 1
        M0[p + 1, 1:2:(p + 1)] .= 1
        M0[p + 1, 2:2:(p + 1)] .= -1
        M1 = Matrix(ToeplitzMatrices.Toeplitz([1; cumprod(wl ./ (1:p))], I[:, 1]))
        M1[p, :] .= ewl
        M1[p + 1, 1:2:(p + 1)] .= ewlm
        M1[p + 1, 2:2:(p + 1)] .= -ewlm

    else
        M0 = I[:, :]
        ww = [1 cumprod(repeat([w * w], 1, m); dims=2)]
        M = ToeplitzMatrices.Toeplitz([1; cumprod(l ./ (1:(p + 2 * m)))], I[:, 1])
        M1 = M[1:(p + 1), :]
        M1[p, :] = ww * M[p:2:end, :]
        M1[p + 1, :] = ww * M[(p + 1):2:end, :]
    end
    cs = zeros(Float64, p + 1)
    C = zeros(Float64, p + 1, p + 1)
    C[p + 1, :] = [M1[:, 1] M0[:, 1:p]]' \ I[:, 1]
    for i in 2:p
        cs = cs + C[p + 1 - i + 2, :]
        cc = zeros(Float64, p + 1)
        cc[i] = -cs' * M1[:, i]
        C[p + 1 - i + 1, :] .= [M1[:, 1:i] M0[:, 1:(p + 1 - i)]]' \ cc
    end
    C[1, :] .= [M0[:, 1] M1[:, 1:p]]' \ I[:, 1]

    return C
end

"""
    get_derivative_space(ect_space::GeneralizedExponential)

Get the space of one degree lower than the input space. Assumes that the degree of the space
is at least 2.

# Arguments
- `ect_space::GeneralizedExponential`: A generalized exponential space.

# Returns
- `::GeneralizedExponential`: A generalized exponential space of one degree lower than the input space.
"""
function get_derivative_space(ect_space::GeneralizedExponential)
    if ect_space.p < 2
        throw(ArgumentError("Degree of the space must be at least 2 to get derivative space."))
    end
    return GeneralizedExponential(
        ect_space.p - 1, ect_space.w, ect_space.l, ect_space.t, ect_space.m
    )
end

"""
    get_child_canonical_space(ect_space::GeneralizedExponential)

Bisect the canonical space by dividing the length in half.

# Arguments
- `ect_space::GeneralizedExponential`: A generalized exponential space.

# Returns
- `::GeneralizedExponential`: A generalized exponential space with the length divided by 2.
"""
function get_bisected_canonical_space(ect_space::GeneralizedExponential)
    return GeneralizedExponential(ect_space.p, ect_space.w, ect_space.l / 2, ect_space.m)
end

"""
    get_child_canonical_space(ect_space::GeneralizedExponential, num_sub_elements::Int)

For number of sub-elements which is powers of 2, bisect the canonical space by dividing the
length in half for each power.

# Arguments
- `ect_space::GeneralizedExponential`: A generalized exponential space.
- `num_sub_elements::Int`: Number of sub-elements to be created.

# Returns
- `::GeneralizedExponential`: A generalized exponential space with the subdivided length.
"""
function get_child_canonical_space(ect_space::GeneralizedExponential, num_sub_elements::Int)
    num_ref = log2(num_sub_elements)
    if num_sub_elements < 2 || !isapprox(num_ref - round(num_ref), 0.0; atol=1e-12)
        throw(
            ArgumentError(
                "Number of subdivisions should be a power of 2 and greater than 1"
            ),
        )
    end

    return GeneralizedExponential(
        ect_space.p, ect_space.w, ect_space.l / num_sub_elements, ect_space.m
    )
end
