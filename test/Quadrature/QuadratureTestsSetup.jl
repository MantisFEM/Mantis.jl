# Helper file which defines a few test functions for quadrature rules.

# Tolerance for comparing floating point numbers.
atol = 1e-12

# Monomial and integrated monomial functions.
function monomial(degree::Int, x::Float64)
    return x^degree
end

function monomial(degree::Int, x::Vector{Float64})
    return x .^ degree
end

function integrated_monomial(degree::Int, x::Float64)
    return x^(degree+1)/(degree+1)
end

function integrated_monomial(degree::Int, x::Vector{Float64})
    return x .^ (degree+1) ./ (degree+1)
end

# Chebyshev and integrated Chebyshev functions.
# This is a recursive function. To reduce the number of recursive calls, we have hardcoded a
# few Chebyshev polynomials.
function chebyshev(degree::Int, x::Float64)
    if degree == 0
        return 1.0
    elseif degree == 1
        return x
    elseif degree == 2
        return 2.0*x^2 - 1.0
    elseif degree == 5
        return 16.0*x^5 - 20.0*x^3 + 5.0*x
    elseif degree == 10
        return 512.0*x^10 - 1280.0*x^8 + 1120.0*x^6 - 400.0*x^4 + 50.0*x^2 - 1.0
    else
        return 2.0*x*chebyshev(degree-1, x) - chebyshev(degree-2, x)
    end
end

function chebyshev(degree::Int, x::Vector{Float64})
    if degree == 0
        return ones(Float64, length(x))
    elseif degree == 1
        return x
    elseif degree == 2
        return @. 2.0*x^2 - 1.0
    elseif degree == 5
        return @. 16.0*x^5 - 20.0*x^3 + 5.0*x
    elseif degree == 10
        return @. 512.0*x^10 - 1280.0*x^8 + 1120.0*x^6 - 400.0*x^4 + 50.0*x^2 - 1.0
    else
        return 2.0 .* x .* chebyshev(degree-1, x) .- chebyshev(degree-2, x)
    end
end

function integrated_chebyshev(degree::Int, x::Float64)
    if degree == 0
        return x
    elseif degree == 1
        return 0.5 * x^2
    else
        return degree * chebyshev(degree+1, x) / (degree^2 - 1) -
               x * chebyshev(degree, x) / (degree - 1)
    end
end

function integrated_chebyshev(degree::Int, x::Vector{Float64})
    if degree == 0
        return x
    elseif degree == 1
        return 0.5 .* x .^ 2
    else
        return degree .* chebyshev(degree+1, x) ./ (degree^2-1) .-
               x .* chebyshev(degree, x) ./ (degree-1)
    end
end

function chebyshev_nd(degrees::NTuple{N, Int}, x::NTuple{N, Vector{Float64}}) where {N}
    cheb_1d = [chebyshev(degrees[i], x[i]) for i in 1:N]
    result = Vector{Float64}(undef, prod(size.(cheb_1d, 1)))
    for (linear_idx, poly_all) in enumerate(Iterators.product(cheb_1d...))
        result[linear_idx] = prod(poly_all)
    end

    return result
end

function integrated_chebyshev_nd(degrees::NTuple{N, Int}) where {N}
    return prod([
        integrated_chebyshev(degrees[i], 1.0) - integrated_chebyshev(degrees[i], 0.0) for
        i in 1:N
    ])
end
