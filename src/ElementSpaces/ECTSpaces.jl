abstract type AbstractECTSpaces <: AbstractElementSpace end

@doc raw"""
    struct GeneralizedTrigonometric <: AbstractECTSpaces

Concrete type for Generalized Trignometric section space spanned by `<1, x, ..., x^(p-2), cos(wx), sin(wx)>` on `[0,1]`.

# Fields
- `p::Int`: Degree of the space.
- `w::Float64`: Weight parameter for the space.
- `t::Bool`: flag to indicate if critical length is exceeded.
- `m::Int`: number of terms from the infinite sum used to build the basis.
- `C::Array{Float64}`: representation matrix for the local basis.
"""
struct GeneralizedTrigonometric <: AbstractECTSpaces
    p::Int
    w::Float64
    t::Bool
    m::Int
    C::Array{Float64}
    endpoint_tol::Float64
    function GeneralizedTrigonometric(p::Int)
        w = 1.0
        t = false
        m = 10
        GeneralizedTrigonometric(p, w, t, m)
    end

    function GeneralizedTrigonometric(p::Int, w::Float64)
        t = abs(w) >= 3.0
        m = 10
        GeneralizedTrigonometric(p, w, t, m)
    end

    function GeneralizedTrigonometric(p::Int, w::Float64, t::Bool)
        m = 10
        GeneralizedTrigonometric(p, w, t, m)
    end

    function GeneralizedTrigonometric(p::Int, w::Float64, t::Bool, m::Int)
        endpoint_tol = 1e-12
        new(p, w, t, m, gtrig_representation(p, w, t, m),endpoint_tol)
     end
end

@doc raw"""
    evaluate(gtrig::GeneralizedTrigonometric, ξ::Vector{Float64}, nderivatives::Int64)::Array{Float64}

Compute derivatives up to order `nderivatives` for all basis functions of degree `p` at `ξ` for ``\xi \in [0.0, 1.0]``. 

# Arguments
- `gtrig::GeneralizedTrigonometric`: Generalized Trigonometric section space.
- `ξ::Vector{Float64}`: vector of evaluation points ``\in [0.0, 1.0]``.
- `nderivatives::Int64`: maximum order of derivatives to be computed (nderivatives ``\leq p``).

See also [`evaluate(gtrig::GeneralizedTrigonometric, ξ::Float64, nderivatives::Int64)`](@ref).
"""
function evaluate(gtrig::GeneralizedTrigonometric, xi::Vector{Float64}, nderivatives::Int)::Array{Float64}
    neval = length(xi)
    ders = zeros(Float64, neval, gtrig.p + 1, nderivatives + 1)
    for i = 1:neval
        ders[i,:,:] = evaluate(gtrig, xi[i],nderivatives)
    end
    return ders
end

function evaluate(gtrig::GeneralizedTrigonometric, xi::Float64, nderivatives::Int)
    tol = gtrig.endpoint_tol
    M = zeros(Float64, 1, gtrig.p+1, nderivatives+1)
    j = xi >= 0.0 && xi < 1.0
    if abs(xi-1.0) < tol
       j = true
    end
    if j
        if gtrig.t
            for r = 0:nderivatives
                k = min(r, gtrig.p-1)
                wxl = gtrig.w * xi
                E = [1.0; cumprod((1.0 ./ (1:gtrig.p-k)) * wxl)]
                if mod(r, 4) == 0
                    E[gtrig.p-k] = cos(wxl);
                    E[gtrig.p+1-k] = sin(wxl);
                elseif mod(r, 4) == 1
                    E[gtrig.p-k] = -sin(wxl);
                    E[gtrig.p+1-k] = cos(wxl);
                elseif mod(r, 4) == 2
                    E[gtrig.p-k] = -cos(wxl);
                    E[gtrig.p+1-k] = -sin(wxl);
                elseif mod(r, 4) == 3
                    E[gtrig.p-k] = sin(wxl);
                    E[gtrig.p+1-k] = -cos(wxl);
               end
                M[1, :, r+1] = (gtrig.w^r) * (gtrig.C[:,k+1:end] * E)
            end
        else
            for r = 0:nderivatives
                k = min(r, gtrig.p-1)
                ww = [1; cumprod(repeat([-gtrig.w * gtrig.w], gtrig.m))]
                Ef = [1.0; cumprod((1.0 ./ (1:gtrig.p-k+2*gtrig.m)) * xi)]
                E = Ef[1:gtrig.p+1-k]
                E[gtrig.p-k, :] = Ef[gtrig.p-k:2:end, :]' * ww
                E[gtrig.p-k+1, :] = Ef[gtrig.p-k+1:2:end, :]' * ww
                M[1, :, r+1] = gtrig.C[:,k+1:end] * E
            end
        end
    end
    return M
end

@doc raw"""
    evaluate(gtrig::GeneralizedTrigonometric, ξ::Vector{Float64})::Array{Float64}

Compute all basis function values at `ξ` in ``[0.0, 1.0]``.

# Arguments
- `gtrig::GeneralizedTrigonometric`:  Generalized Trigonometric section space.
- `xi::Vector{Float64}`: vector of evaluation points ``\in [0.0, 1.0]``.

See also [`evaluate(gtrig::GeneralizedTrigonometric, xi::Vector{Float64}, nderivatives::Int64)`](@ref).
"""
function evaluate(gtrig::GeneralizedTrigonometric, xi::Vector{Float64})
    return evaluate(gtrig, xi, 0)
end

function evaluate(gtrig::GeneralizedTrigonometric, xi::Float64)
    return evaluate(gtrig, xi, 0)
end


@doc raw"""
    gtrig_representation(p::Int, w::Float64, t::Bool, m::Int)

Build representation matrix for Generalized Trignometric section space of degree `p`, weight `w` and `m` terms.

# Arguments
- `p::Int`: Degree of the space.
- `w::Float64`: Weight parameter for the space.
- `t::Bool`: flag to indicate if critical length is exceeded.
- `m::Int`: number of terms from the infinite sum used to build the basis.

# Returns:
- `C::Array{Float64}`: representation matrix for the local basis.
"""

import LinearAlgebra, ToeplitzMatrices

function gtrig_representation(p::Int, w::Float64, t::Bool, m::Int)
    
    I = Matrix(1.0LinearAlgebra.I, p+1, p+1)
    if t
        cw = cos(w)
        sw = sin(w)
        M0 = I[:,:]
        M0[[p, p+1], [p, p+1]] = 0.0
        M0[p, 1:4:end] .= 1
        M0[p, 3:4:end] .= -1
        M0[p+1, 2:4:end] .= 1
        M0[p+1, 4:4:end] .= -1
        M1 = ToeplitzMatrices.Toeplitz([1; cumprod(w ./ (1:p))], reshape(I, ((p+1)^2,)))
        M1[p, 1:4:end] .= cw
        M1[p, 2:4:end] .= -sw
        M1[p, 3:4:end] .= -cw
        M1[p, 4:4:end] .= sw
        M1[p+1, 1:4:end] .= sw
        M1[p+1, 2:4:end] .= cw
        M1[p+1, 3:4:end] .= -sw
        M1[p+1, 4:4:end] .= -cw
    else
        M0 = I[:,:]
        ww = [1 cumprod(repeat([-w * w], 1, m), dims=2)]
        M = ToeplitzMatrices.Toeplitz([1; cumprod(1.0 ./ (1:p+2*m))], reshape(I, ((p+1)^2,)))
        M1 = M[1:p+1, :]
        M1[p, :] = ww * M[p:2:end, :]
        M1[p+1, :] = ww * M[p+1:2:end, :]
    end
    cs = zeros(Float64, p+1)
    C = zeros(Float64, p+1, p+1)
    C[p+1, :] =  [M1[:, 1] M0[:, 1:p]]' \ [1.0; zeros(p)]
    for i = 2:p
        cs = cs + C[p+1-i+2, :]
        cc = zeros(Float64, p+1)
        cc[i] = -cs' * M1[:, i]
        C[p+1-i+1, :] = [M1[:, 1:i] M0[:, 1:p+1-i]]' \ cc
    end
    C[1, :] = [M0[:, 1] M1[:, 1:p]]' \ [1.0; zeros(p)] 

    return C
end