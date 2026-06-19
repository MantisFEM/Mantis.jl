# # Quadrature and integrating a form

# Every integral `Mantis` computes (mass and stiffness matrices, load vectors, error norms) is
# evaluated numerically with a *quadrature rule*: a set of points and weights. This example
# shows how to build a rule, integrate a form with it, and why the number of points matters.

using Mantis

# ## Setting up

# We work on the unit interval and integrate the simple ``0``-form ``f(x) = x^2``. The integral
# operator ``\int`` pairs forms via ``f \wedge \star f``, so integrating ``f \wedge \star f``
# computes ``\int_0^1 f(x)^2\,\mathrm{d}x = \int_0^1 x^4\,\mathrm{d}x = \tfrac{1}{5}``.

geometry = Geometry.create_cartesian_box((0.0,), (1.0,), (4,))
f = Forms.AnalyticalFormField(0, x -> [x[:, 1] .^ 2], geometry, "f")

exact = 1 / 5

# ## Building a global quadrature rule

# A global rule places a *canonical* rule on every element. The canonical rule is a
# [`tensor_product_rule`](@ref Mantis.Quadrature.tensor_product_rule) of a univariate rule (here
# Gauss-Legendre) with a chosen number of points per direction. A Gauss-Legendre rule
# with `q` points integrates polynomials up to degree ``2q - 1`` exactly.

num_elements = Geometry.get_num_elements(geometry)

# This helper integrates `f ∧ ★f` with `q` Gauss points per element and returns the total.
function integrate_f_squared(q)
    canonical_rule = Quadrature.tensor_product_rule((q,), Quadrature.gauss_legendre)
    dΩ = Quadrature.StandardQuadrature(canonical_rule, num_elements)
    integrand = ∫(f ∧ ★(f), dΩ)
    return sum(Forms.evaluate(integrand, e)[1][1] for e in 1:num_elements)
end

# ## Too few points vs. enough points

# Our integrand ``x^4`` is degree 4, so it needs ``2q - 1 \ge 4``, i.e. ``q \ge 3`` points for
# *exact* integration. With one or two points the result is wrong; with three it is exact:

for q in 1:4
    approx = integrate_f_squared(q)
    println("q = $q point(s)/element:  ∫ ≈ $(round(approx; digits=8))   (error ",
            round(abs(approx - exact); sigdigits=2), ")")
end

println("\nexact value ∫₀¹ x⁴ dx = ", exact)

# The take-away: choose the quadrature degree from the polynomial degree of your integrand.
# For a B-spline mass/stiffness matrix of degree `p`, a Gauss rule with `p + 1` points per
# direction is the standard choice, which is why the PDE examples use
# `Quadrature.tensor_product_rule(p .+ 1, Quadrature.gauss_legendre)`. For *error* computation
# it is good practice to use a still-finer rule, so the quadrature error does not pollute the
# measured discretisation error (see the [L2 projection](@ref) example).
