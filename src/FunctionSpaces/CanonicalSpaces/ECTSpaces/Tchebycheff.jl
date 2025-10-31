"""
    Tchebycheff <: AbstractECTSpaces

A Tchebycheffian section space. The parameters are the roots of a differential operator with
constant coefficients.
* A complex root `α + iβ`, `β ≠ 0`, of multiplicity `m` contributes `2m` basis functions of
  the form (upto scaling):
    `x^i e^(αx) cos(βx)`, `x^i e^(αx) sin(βx)`, `i = 0, ..., m-1`
* A real root `α` of multiplicity `m` contributes `m` basis functions of the
  form (upto scaling):
    `x^i e^(αx)`, `i = 0, ..., m-1`

# Fields
- `p::Int`: Degree of the space.
- `roots::Matrix{Float64}`: (real, imag) pairs of roots.
- `root_mult::Vector{Int}`: Multiplicities of roots.
- `root_type::Vector{Int}`: Types of roots: 0=zero, 1=real, 2=imaginary, 3=complex.
- `l::Float64`: Length of the interval. Tchebycheff space is not scale-invariant.
- `mu::Vector{Int}`: Cumulative dimension for the roots.
- `C::Matrix{Float64}`: Representation matrix for the local basis.
- `endpoint_tol::Float64`: Tolerance to determine if a point is at an endpoint.
"""
struct Tchebycheff <: AbstractECTSpaces
    p::Int
    roots::Matrix{Float64} # (real, imag) pairs of roots
    root_mult::Vector{Int} # multiplicities of roots
    root_type::Vector{Int} # types of roots: 0=zero, 1=real, 2=imaginary, 3=complex
    l::Float64
    mu::Vector{Int} # cumulative dimension
    C::Matrix{Float64} # representation matrix
    endpoint_tol::Float64

    function Tchebycheff(
        p::Int,
        roots::Matrix{Float64} = [0.0 0.0],
        l::Float64 = 1.0,
        root_mult::Vector{Int} = ones(Int, size(roots, 1)),
        endpoint_tol::Float64 = 1e-12,
    )
        if any(root_mult .< 1)
            throw(ArgumentError("All multiplicities must be positive integers."))
        end
        if size(roots, 1) != length(root_mult)
            throw(ArgumentError("The number of roots must match the number of multiplicities."))
        end

        # make all imaginary parts non-negative
        roots[:, 2] .= abs.(roots[:, 2])
        # total multiplicity: each real root counts once, each complex root counts twice
        m = sum(((roots[:,2] .!= 0) .+ 1) .* root_mult)
        if p+1 != m
            if p+1 > m
                roots = vcat(roots, [0.0 0.0])
                push!(root_mult, p+1 - m)
            else
                throw(ArgumentError("The local dimension, $(p+1), must be greater than or equal to the number of roots (counting multiplicities), $(m)."))
            end
        end

        # find unique roots and their indices
        roots_i = unique(i -> roots[i, :], 1:size(roots,1))
        roots_unq = roots[roots_i, :]
        # find multiplicities of unique roots
        root_mult_unq = similar(root_mult[roots_i])
        count = 0
        for ru in eachrow(roots_unq)
            root_mult_unq[count+1] = sum(root_mult[findall(row -> all(row .== ru), eachrow(roots))])
            count += 1
        end
        # store types of roots
        root_type = zeros(Int, length(root_mult_unq))
        root_type[findall(roots_unq[:,1] .!= 0.0)] .= 1
        root_type[findall(roots_unq[:,2] .!= 0.0)] .= 2
        root_type[findall(all(roots_unq .!= 0, dims=2))] .= 3

        # number of basis functions for each unique root
        mu = [0, cumsum(((roots_unq[:, 2] .!= 0) .+ 1) .* root_mult_unq)...]

        return new(
            p,
            roots_unq,
            root_mult_unq,
            root_type,
            l,
            mu,
            tcheb_representation(p, roots_unq, root_mult_unq, root_type, l, mu),
            endpoint_tol
        )
    end
end

function _evaluate(ect_space::Tchebycheff, xi::Float64, nderivatives::Int)
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
    for r = 0:nderivatives
        zero_v = zeros(1)
        E = zeros(ect_space.p+1)
        m = maximum(ect_space.root_mult) - 1
        X = vcat([1.0], cumprod((xi ./ (1:m))))
        for i in eachindex(ect_space.root_mult)
            if ect_space.root_type[i] == 0
                Ei = zeros(ect_space.root_mult[i])
                Ei[r+1:end] .= X[1:ect_space.root_mult[i]-r]
            elseif ect_space.root_type[i] == 1
                ewxl = exp(ect_space.roots[i, 1] * xi)
                Ei = X[1:ect_space.root_mult[i]] .* ewxl
                for _ in 1:r
                    Ei .= vcat(zero_v, Ei[1:end-1]) .+ ect_space.roots[i, 1] * Ei
                end
            elseif ect_space.root_type[i] == 2
                cwxl = cos(ect_space.roots[i, 2] * xi)
                swxl = sin(ect_space.roots[i, 2] * xi)
                Eci = X[1:ect_space.root_mult[i]] .* cwxl
                Esi = X[1:ect_space.root_mult[i]] .* swxl
                for _ in 1:r
                    Ezi = copy(Eci)
                    Eci .= vcat(zero_v, Eci[1:end-1]) .- ect_space.roots[i, 2] * Esi
                    Esi .= vcat(zero_v, Esi[1:end-1]) .+ ect_space.roots[i, 2] * Ezi
                end
                Ei = vcat(Eci, Esi)
            elseif ect_space.root_type[i] == 3
                ewxl = exp(ect_space.roots[i, 1] * xi)
                ecwxl = ewxl * cos(ect_space.roots[i, 2] * xi)
                eswxl = ewxl * sin(ect_space.roots[i, 2] * xi)
                Eci = X[1:ect_space.root_mult[i]] .* ecwxl
                Esi = X[1:ect_space.root_mult[i]] .* eswxl
                for _ in 1:r
                    Ezi = copy(Eci)
                    Eci .= vcat(zero_v, Eci[1:end-1]) .+ ect_space.roots[i, 1] * Eci .- ect_space.roots[i, 2] * Esi
                    Esi .= vcat(zero_v, Esi[1:end-1]) .+ ect_space.roots[i, 1] * Esi .+ ect_space.roots[i, 2] * Ezi
                end
                Ei = vcat(Eci, Esi)
            end
            E[ect_space.mu[i]+1:ect_space.mu[i+1]] .= Ei
        end
        # rescale the derivative to map back from [0, l] -> [0, 1]
        M[1, :, r+1] = ect_space.C * E * (ect_space.l^r)
    end

    if left
        M[1, :, :] = LinearAlgebra.triu(M[1, :, :])
    elseif right
        M[1, :, :] = LinearAlgebra.triu!(M[1, end:-1:1, :])[end:-1:1, :]
    end

    return M
end

"""
    evaluate(ect_space::Tchebycheff, ξ::Vector{Float64})

Compute all basis function values at `ξ` in ``[0.0, 1.0]``.

# Arguments
- `ect_space::Tchebycheff`:  Tchebycheff section space.
- `xi::Vector{Float64}`: vector of evaluation points ``in [0.0, 1.0]``.
"""
function evaluate(ect_space::Tchebycheff, xi::Vector{Float64})
    return evaluate(ect_space, xi, 0)
end

function evaluate(ect_space::Tchebycheff, xi::Float64)
    return evaluate(ect_space, [xi], 0)
end

"""
    tcheb_representation(p::Int, roots::Matrix{Float64}, root_mult::Vector{Int}, root_type::Vector{Int},
                        l::Float64, mu::Vector{Int})

Build representation matrix for Tchebycheff section space of degree `p`, roots `roots`, root multiplicities
`root_mult`, root types `root_type`, length `l`, and cumulative dimensions `mu`.

# Arguments
- `p::Int`: Degree of the space.
- `roots::Matrix{Float64}`: (real, imag) pairs of roots.
- `root_mult::Vector{Int}`: Multiplicities of roots.
- `root_type::Vector{Int}`: Types of roots: 0=zero, 1=real, 2=imaginary, 3=complex.
- `l::Float64`: Length of the interval.
- `mu::Vector{Int}`: Cumulative dimension for the roots.

# Returns
- `C::Matrix{Float64}`: Representation matrix for the local basis.
"""
function tcheb_representation(
    p::Int,
    roots::Matrix{Float64},
    root_mult::Vector{Int},
    root_type::Vector{Int},
    l::Float64,
    mu::Vector{Int},
)
    M0 = zeros(p+1, p+1)
    M1 = zeros(p+1, p+1)
    m = maximum(root_mult) - 1
    X = vcat([1.0], cumprod(l ./ (1:m)))
    I = 1.0LinearAlgebra.I
    for i in eachindex(root_mult)
        r_ind = root_type[i]
        idx = (mu[i]+1):mu[i+1]
        if r_ind == 0
            M0i = Matrix(I, root_mult[i], p+1)
            M1i = Matrix(ToeplitzMatrices.Toeplitz(X[1:root_mult[i]], [1.0; zeros(p)]))
        elseif r_ind == 1
            ewl = exp(roots[i, 1] * l)
            M0i = Matrix(I, root_mult[i], p+1)
            M1i = zeros(root_mult[i], p+1)
            M1i[:, 1] .= X[1:root_mult[i]] * ewl
            for k in 1:p
                ki = min(k, root_mult[i]) - 1
                M0i[1:ki+1, k+1] .= vcat(0.0, M0i[1:ki, k]) + roots[i, 1] * M0i[1:ki+1, k]
                M1i[:, k+1] .= vcat(0.0, M1i[1:end-1, k]) + roots[i, 1] * M1i[:, k]
            end
        elseif r_ind == 2
            cwl = cos(roots[i, 2] * l)
            swl = sin(roots[i, 2] * l)
            n4 = root_mult[i]
            M0ci = Matrix(I, n4, p+1)
            M0si = zeros(n4, p+1)
            M1ci = zeros(n4, p+1)
            M1si = zeros(n4, p+1)
            M1ci[:, 1] .= X[1:n4] * cwl
            M1si[:, 1] .= X[1:n4] * swl
            for k in 1:p
                ki = min(k, n4) - 1
                M0ci[1:ki+1, k+1] .= vcat(0.0, M0ci[1:ki, k]) - roots[i, 2] * M0si[1:ki+1, k]
                M0si[1:ki+1, k+1] .= vcat(0.0, M0si[1:ki, k]) + roots[i, 2] * M0ci[1:ki+1, k]
                M1ci[:, k+1] .= vcat(0.0, M1ci[1:end-1, k]) - roots[i, 2] * M1si[:, k]
                M1si[:, k+1] .= vcat(0.0, M1si[1:end-1, k]) + roots[i, 2] * M1ci[:, k]
            end
            M0i = vcat(M0ci, M0si)
            M1i = vcat(M1ci, M1si)
        elseif r_ind == 3
            ecwl = exp(roots[i, 1] * l) * cos(roots[i, 2] * l)
            eswl = exp(roots[i, 1] * l) * sin(roots[i, 2] * l)
            n4 = root_mult[i]
            M0ci = Matrix(I, n4, p+1)
            M0si = zeros(n4, p+1)
            M1ci = zeros(n4, p+1)
            M1si = zeros(n4, p+1)
            M1ci[:, 1] .= X[1:n4] * ecwl
            M1si[:, 1] .= X[1:n4] * eswl
            for k in 1:p
                ki = min(k, n4) - 1
                M0ci[1:ki+1, k+1] .= vcat(0.0, M0ci[1:ki, k]) +
                    roots[i, 1] * M0ci[1:ki+1, k] - roots[i, 2] * M0si[1:ki+1, k]
                M0si[1:ki+1, k+1] .= vcat(0.0, M0si[1:ki, k]) +
                    roots[i, 1] * M0si[1:ki+1, k] + roots[i, 2] * M0ci[1:ki+1, k]
                M1ci[:, k+1] .= vcat(0.0, M1ci[1:end-1, k]) +
                    roots[i, 1] * M1ci[:, k] - roots[i, 2] * M1si[:, k]
                M1si[:, k+1] .= vcat(0.0, M1si[1:end-1, k]) +
                    roots[i, 1] * M1si[:, k] + roots[i, 2] * M1ci[:, k]
            end
            M0i = vcat(M0ci, M0si)
            M1i = vcat(M1ci, M1si)
        end
        M0[idx, :] .= M0i
        M1[idx, :] .= M1i
    end
    cs = zeros(p+1)
    C = zeros(p+1, p+1)
    # Solve for last row
    C[p+1, :] = (Matrix(I, 1, p+1)) / hcat(M1[:, 1], M0[:, 1:p])
    for i in 2:p
        cs .+= C[p+1-i+2, :]
        cc = zeros(1, p+1)
        cc[i] = reduce(+, -cs .* M1[:, i])
        C[p+1-i+1, :] = cc / hcat(M1[:, 1:i], M0[:, 1:p+1-i])
    end
    C[1, :] = (Matrix(I, 1, p+1)) / hcat(M0[:, 1], M1[:, 1:p])

    return C
end

"""
    get_derivative_space(ect_space::Tchebycheff)

Get the space of one degree lower than the input space. Requires that the input space has
the zero root with multiplicity at least 1.

# Arguments
- `ect_space::Tchebycheff`: A Tchebycheff space.

# Returns
- `::Tchebycheff`: A Tchebycheff space of one degree lower than the input space.
"""
function get_derivative_space(ect_space::Tchebycheff)
    zero_i = findall(all(ect_space.roots .== 0.0, dims=2))
    if length(zero_i) == 0
        throw(ArgumentError("The zero root must exist for the input Tchebycheff space."))
    end
    if ect_space.root_mult[zero_i[1]] == 1
        idx = setdiff(1:length(ect_space.root_mult), zero_i)
        new_roots = ect_space.roots[idx, :]
        new_root_mult = ect_space.root_mult[idx]
        new_root_type = ect_space.root_type[idx]
    else
        new_roots = ect_space.roots
        new_root_mult = copy(ect_space.root_mult)
        new_root_mult[zero_i[1]] -= 1
        new_root_type = copy(ect_space.root_type)
    end
    return Tchebycheff(
        ect_space.p-1, new_roots, ect_space.l, new_root_mult
    )
end

"""
    get_bisected_canonical_space(ect_space::Tchebycheff)

Bisect the canonical space by dividing the length in half.

# Arguments
- `ect_space::Tchebycheff`: A Tchebycheff space.

# Returns
- `::Tchebycheff`: A Tchebycheff space with the length divided by 2.
"""
function get_bisected_canonical_space(ect_space::Tchebycheff)
    return Tchebycheff(ect_space.p, ect_space.roots, ect_space.l/2, ect_space.root_mult)
end

"""
    get_finer_canonical_space(ect_space::Tchebycheff, num_sub_elements::Int)

Bisect the canonical space by dividing the length in half for each power.

# Arguments
- `ect_space::Tchebycheff`: A Tchebycheff space.
- `num_sub_elements::Int`: Number of sub-elements to be created.

# Returns
- `::Tchebycheff`: A Tchebycheff space with the subdivided length.
"""
function get_finer_canonical_space(ect_space::Tchebycheff, num_sub_elements::Int)
    num_ref = log2(num_sub_elements)
    if num_sub_elements < 2 || !isapprox(num_ref - round(num_ref), 0.0; atol=1e-12)
        throw(
            ArgumentError(
                "Number of subdivisions should be a power of 2 and greater than 1"
            ),
        )
    end

    return Tchebycheff(
        ect_space.p, ect_space.roots, ect_space.l/num_sub_elements, ect_space.root_mult
    )
end
