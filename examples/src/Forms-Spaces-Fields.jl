# # Form spaces and form fields

# `Mantis` does not solve PDEs in terms of plain functions; it works with *differential
# forms*. A function space (see [B-spline spaces and basis functions](@ref)) only says *which*
# functions are available; wrapping it in a **form space** additionally says *how the object
# transforms* under a change of coordinates, which is what makes the structure-preserving
# machinery work. This example shows how to turn a function space into form spaces of different
# rank, and how to represent concrete fields on them.

using Mantis

# ## From a function space to form spaces

# We start from a 2D B-spline space on the unit square.

geometry = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (4, 4))
B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (4, 4), (2, 2), (1, 1))

# The same `B` can back several form spaces. The *rank* `k` is the first argument to
# [`FormSpace`](@ref Mantis.Forms.FormSpace). A scalar space like `B` can represent a
# ``0``-form (an ordinary function) or a top (here ``2``-) form (a density that carries the
# Jacobian determinant when pulled back). Both have one component, so both reuse `B` directly:

Λ⁰ = Forms.FormSpace(0, B, "ω⁰")   # a 0-form (function)
Λ² = Forms.FormSpace(2, B, "ω²")   # a 2-form (top form / density)

println("dim(Λ⁰) = ", Forms.get_num_basis(Λ⁰), ",  dim(Λ²) = ", Forms.get_num_basis(Λ²))

# Although they share a basis, `Λ⁰` and `Λ²` use *different pullbacks*, so they evaluate
# differently on a curved geometry and admit different operators; see the
# [Differential-form operators](@ref) example. A ``1``-form in 2D has two components and so
# needs a vector-valued space; the easiest way to get a compatible family of ``0``-, ``1``-
# and ``2``-form spaces at once is the de Rham-complex helper used in the
# [L2 projection](@ref) example.

# ## Form fields: a basis plus coefficients

# A [`FormField`](@ref Mantis.Forms.FormField) is a *specific* form: a form space together with
# a coefficient for each basis function. This is how solutions and right-hand sides are
# represented. Here we build one by hand from a coefficient vector:

n = Forms.get_num_basis(Λ⁰)
coefficients = LinRange(0.0, 1.0, n)          # any vector of length dim(Λ⁰)
u = Forms.build_form_field(Λ⁰, collect(coefficients))

println("\nu has ", Forms.get_num_coefficients(u), " coefficients")

# To evaluate it, hand `evaluate` an element id and points in canonical coordinates; it returns
# the field values (and the basis indices, which we ignore here):

points = Points.CartesianPoints(([0.0, 0.5, 1.0], [0.5]))
values, _ = Forms.evaluate(u, 1, points)
println("u on element 1 at three points: ", values[1])

# ## Analytical form fields: a closed-form expression

# When you already know a field in closed form (an exact solution, a forcing term), use an
# [`AnalyticalFormField`](@ref Mantis.Forms.AnalyticalFormField). Instead of coefficients it
# stores a function of the physical coordinates (a `Matrix` with one row per point, one column
# per dimension), returning one vector per form component:

f = Forms.AnalyticalFormField(0, x -> [sin.(2π .* x[:, 1]) .* x[:, 2]], geometry, "f")

f_values, _ = Forms.evaluate(f, 1, points)
println("f on element 1 at three points: ", f_values[1])

# `FormField`s and `AnalyticalFormField`s interoperate: you can subtract them (e.g. to form an
# error `uₕ - uₑ`), feed them to operators, and integrate them, which is what the
# [L2 projection](@ref) and [Biharmonic](@ref) examples do.
