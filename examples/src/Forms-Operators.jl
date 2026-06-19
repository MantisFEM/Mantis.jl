# # Differential-form operators

# `Mantis` works with forms (see [Form spaces and form fields](@ref)) because they carry a
# small set of operators: the exterior derivative ``\mathrm{d}``, the wedge ``\wedge``, and the
# Hodge star ``\star``. These express the differential operators of vector calculus in a way
# that does not depend on the dimension. This example shows how the operators move between form
# ranks, and uses them to assemble the building block of the Laplacian.

using Mantis

# We build a compatible family of ``0``-, ``1``- and ``2``-form spaces (a discrete de Rham
# complex) on the unit square, and a generic ``0``-form field on it.

X = Forms.create_tensor_product_bspline_de_rham_complex(
    (0.0, 0.0), (1.0, 1.0), (4, 4), (2, 2), (1, 1)
)
Λ⁰, Λ¹, Λ² = X[1], X[2], X[3]
geometry = Forms.get_geometry(Λ⁰)
num_elements = Geometry.get_num_elements(geometry)

u = Forms.build_form_field(Λ⁰, collect(LinRange(-1.0, 1.0, Forms.get_num_basis(Λ⁰))))

# ## The exterior derivative raises the rank

# Applied to a ``0``-form, ``\mathrm{d}`` is the gradient; to a ``1``-form, the (scalar) curl;
# to a ``2``-form in 2D, it vanishes. Each application raises the form rank by one. You can
# read the rank of any form expression with `get_form_rank`:

du = d(u)        # gradient of u: a 1-form

println("rank(u)   = ", Forms.get_form_rank(u))
println("rank(d u) = ", Forms.get_form_rank(du))

# ## ``\mathrm{d} \circ \mathrm{d} = 0``

# Composing ``\mathrm{d}`` with itself raises the rank twice, so `d(d(u))` is structurally a
# ``2``-form, but its value is identically zero. This identity is central to Finite Element
# Exterior Calculus, and because the spaces above form a compatible (exact) complex it holds to
# round-off for the discrete fields, not just in the continuous setting:

println("rank(d(d u)) = ", Forms.get_form_rank(d(du)), "  (but d(d u) ≡ 0)")

# Because `d(d(u))` is zero by construction, `Mantis` does not implement evaluating it. The
# kernel and image structure of the complex is what makes the discretisations in the
# [Maxwell eigenvalue problem](@ref) example well-posed.

# ## Hodge star and wedge change rank predictably

# The Hodge star maps a ``k``-form to an ``(n-k)``-form (here ``n = 2``), and the wedge of a
# ``k``-form with an ``l``-form is a ``(k+l)``-form. Reading off the ranks confirms the
# bookkeeping `Mantis` does for you when you write a weak form:

println("\nrank(★ u)         = ", Forms.get_form_rank(★(u)), "   (n - 0 = 2)")
println("rank(★ d u)       = ", Forms.get_form_rank(★(du)), "   (n - 1 = 1)")
println("rank(d u ∧ ★ d u) = ", Forms.get_form_rank(du ∧ ★(du)), "   (1 + 1 = 2)")

# ## From operators to a number: the Dirichlet energy

# The combination ``\mathrm{d}u \wedge \star\,\mathrm{d}u`` is a top (``2``-)form, so it can be
# integrated to a scalar: the ``H^1`` semi-norm (Dirichlet energy) of `u`. This is the integrand
# of the ``0``-form Laplacian's bilinear form `∫(d(v) ∧ ★(d(u)), dΩ)` from the
# [Hodge Laplacian](@ref) example, here evaluated with `u` on both sides:

canonical_rule = Quadrature.tensor_product_rule((3, 3), Quadrature.gauss_legendre)
dΩ = Quadrature.StandardQuadrature(canonical_rule, num_elements)

energy = sum(Forms.evaluate(∫(du ∧ ★(du), dΩ), e)[1][1] for e in 1:num_elements)
println("\nDirichlet energy  ∫ d u ∧ ★ d u  = ", energy)

# The same three operators (`d`, `∧`, `★`) described on the
# [Finite Element Exterior Calculus (FEEC)](@ref) theory page are what you write down,
# unchanged, to assemble a finite element problem.
