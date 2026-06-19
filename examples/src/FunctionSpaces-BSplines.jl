# # B-spline spaces and basis functions

# A *function space* provides the basis functions you approximate a solution with. `Mantis`'s
# default building block is the **B-spline space**, characterised by two numbers per direction:
# the polynomial **degree** `p` and the inter-element **regularity** `k` (how many derivatives
# are continuous across element boundaries, with ``-1 \le k < p``). This example builds B-spline
# spaces, counts and plots their basis functions, and shows how `p` and `k` change the picture.

using Mantis

# ## A univariate space and its dimension

# We put a B-spline space on a 1D geometry with the constructor
# [`BSplineSpace`](@ref Mantis.FunctionSpaces.BSplineSpace). Here: 4 elements on ``[0, 1]``,
# degree `p = 2`, regularity `k = 1` (so the basis functions are ``C^1``).

line = Geometry.CartesianGeometry((collect(LinRange(0.0, 1.0, 5)),))   # 4 elements
p, k = 2, 1
B = FunctionSpaces.BSplineSpace(line, p, k)

# The dimension of a 1D B-spline space relates to the number of elements `N`, degree `p`, and
# regularity `k` by ``n = N(p+1) - (k+1)(N-1)``. We can check that against `get_num_basis`:

N = Geometry.get_num_elements(line)
formula = N * (p + 1) - (k + 1) * (N - 1)
println("degree p=$p, regularity k=$k, N=$N elements")
println("  dim from formula      = ", formula)
println("  dim from get_num_basis = ", FunctionSpaces.get_num_basis(B))

# Lowering the regularity (more independent functions) raises the dimension; raising it lowers
# it. A few combinations on the same 4-element mesh:

for (pp, kk) in [(2, 0), (2, 1), (3, 2), (3, 0)]
    Bpk = FunctionSpaces.BSplineSpace(line, pp, kk)
    println("  p=$pp, k=$kk → dim = ", FunctionSpaces.get_num_basis(Bpk))
end

# ## Plotting the basis functions

# To draw the individual basis functions we wrap the space in a ``0``-form space and isolate one
# basis function at a time by setting a single coefficient to `1`. (This is the same trick the
# [Heat Equation](@ref) example uses.)

BF  = Forms.FormSpace(0, B, "ϕ")
field = Forms.FormField(BF)
dim = Forms.get_num_basis(BF)

using GLMakie
using DisplayAs #hide

fig = Figure(; size=(650, 350))
ax = Axis(fig[1, 1]; title="B-spline basis (p=$p, k=$k)", xlabel="x", ylabel="ϕᵢ(x)")

ξ = Points.CartesianPoints((collect(LinRange(0.0, 1.0, 30)),))
for i in 1:dim
    fill!(field.coefficients, 0.0)
    field.coefficients[i] = 1.0
    for e in 1:N
        ϕ, _ = Forms.evaluate(field, e, ξ)
        x = Geometry.evaluate(Forms.get_geometry(BF), e, ξ)
        lines!(ax, vec(x), ϕ[1])
    end
end

fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide

# Each function is a bump with local support, non-zero on only a few elements. Away from the
# ends, neighbouring functions are smooth (here ``C^1``) translates of one another. Raising `k`
# makes them smoother; lowering it makes them more localised.

# ## Multivariate spaces

# For boxes there is a one-shot tensor-product helper,
# [`create_bspline_space`](@ref Mantis.FunctionSpaces.create_bspline_space), taking the corner,
# size, element counts, degrees, and regularities as tuples. Its dimension is the product of
# the per-direction dimensions:

B2d = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (4, 4), (2, 2), (1, 1))
println("\n2D space dim = ", FunctionSpaces.get_num_basis(B2d), "  (= 6 × 6)")

# This is the kind of space the PDE examples build on. To make it represent a form, wrap it as
# in [Form spaces and form fields](@ref); to integrate against it, see
# [Quadrature and integrating a form](@ref).
