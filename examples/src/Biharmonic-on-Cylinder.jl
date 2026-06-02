# # Biharmonic problem on a cylindrical surface
#
# This example solves the ``0``-form biharmonic problem on a **cylindrical surface**
# by combining two building blocks from earlier examples:
# - the tensor-product cylinder geometry from [`Tensor-product geometry`](@ref), and
# - the dimension-agnostic biharmonic weak form from [`Biharmonic`](@ref).
#
# The novel ingredient is a **periodic** B-spline space in the angular direction,
# constructed via `GTBSplineSpace`, combined with an explicit cylindrical geometry.
#
# ## Formulation
#
# The cylindrical surface of radius ``R`` is parameterized by
# ``(\theta, z) \in [0, 2\pi] \times [0, H]`` with the identification
# ``\theta = 0 \sim \theta = 2\pi``.
# The induced (pullback) metric is ``g = \mathrm{diag}(R^2, 1)``,
# giving the Laplace--Beltrami operator
#
# ```math
# \Delta_g = \frac{1}{R^2}\frac{\partial^2}{\partial\theta^2}
#            + \frac{\partial^2}{\partial z^2}.
# ```
#
# The ``0``-form biharmonic problem with homogeneous Dirichlet boundary conditions at
# ``z = 0`` and ``z = H`` reads
#
# ```math
# \Delta_g^2\phi = -f \quad \text{on } \Omega, \qquad
# \phi\big|_{\partial\Omega} = 0, \qquad
# \Delta_g\phi\big|_{\partial\Omega} = 0.
# ```
#
# We choose ``R = 1``, ``H = \pi`` and the manufactured solution
#
# ```math
# \phi(\theta, z) = \cos(\theta)\sin(z).
# ```
#
# Since ``\sin(0) = \sin(\pi) = 0``, ``\phi`` vanishes on both boundary circles
# (``z = 0`` and ``z = \pi``) for **all** ``\theta``.
# With ``R = 1`` the Laplace--Beltrami operator gives
#
# ```math
# \Delta_g\phi = -(1 + 1)\,\phi = -2\phi,
# ```
#
# which also vanishes on both boundary circles, satisfying the natural biharmonic
# boundary condition ``\Delta_g\phi|_{\partial\Omega} = 0``.
# Applying ``\Delta_g`` once more gives ``\Delta_g^2\phi = 4\phi``, so the
# forcing function is ``f = -4\cos(\theta)\sin(z)``.
#
# Unlike ``\sin(\theta)``, ``\cos(\theta)`` is **non-zero** at the periodic join
# ``\theta = 0 = 2\pi`` (``\cos(0) = 1``).  This means the boundary-condition
# setup must constrain **only** the ``z = 0`` and ``z = H`` degrees of freedom,
# not the periodic-join degrees of freedom.
#
# ## Implementation
#
# ### Geometry
#
# We build the cylinder surface geometry explicitly as a tensor product of a mapped
# circle curve and a line segment, following the [`Tensor-product geometry`](@ref)
# example.

using Mantis

R   = 1.0
H   = float(π)
n_θ = 16       # elements in the angular direction
n_z = 16       # elements in the axial direction
p   = 3       # B-spline degree (both directions)
k   = 2       # interior regularity (both directions)

# Circle curve: 1-D Cartesian geometry on ``[0, 2\pi]`` mapped to ``\mathbb{R}^2``.

circle_base = Geometry.create_cartesian_box((0.0,), (2π,), (n_θ,))

function ϕ_circle(x::AbstractVector)
    θ = x[1]
    return [R * cos(θ), R * sin(θ)]
end
function dϕ_circle(x::AbstractVector)
    θ = x[1]
    return [-R * sin(θ); R * cos(θ)]
end
function d²ϕ_circle(x::AbstractVector)
    θ = x[1]
    return (
        fill(-R * cos(θ), 1, 1),   # d²(R cosθ)/dθ² = -R cosθ  (1×1 matrix)
        fill(-R * sin(θ), 1, 1),   # d²(R sinθ)/dθ² = -R sinθ  (1×1 matrix)
    )
end

circle = Geometry.MappedGeometry(
    circle_base, Geometry.Mapping((1, 2), ϕ_circle, dϕ_circle, d²ϕ_circle)
)

# Height line.

height_line = Geometry.create_cartesian_box((0.0,), (H,), (n_z,))

# Cylinder surface (2-D manifold in ``\mathbb{R}^3``).

cylinder_geo = Geometry.TensorProductGeometry((circle, height_line))

Geometry.get_manifold_dim(cylinder_geo) == 2
Geometry.get_image_dim(cylinder_geo)    == 3
Geometry.get_num_elements(cylinder_geo) == n_θ * n_z

# ### Function space
#
# #### Angular direction — periodic B-splines
#
# We build a standard B-spline space on ``[0, 2\pi]`` and wrap it with
# `GTBSplineSpace` to identify the endpoints with ``C^k`` smoothness.
# The single entry `[k]` in the regularity vector specifies the smoothness at
# the (unique, periodic) interface.

B_θ  = FunctionSpaces.create_bspline_space(0.0, 2π, n_θ, p, k)
GB_θ = FunctionSpaces.GTBSplineSpace((B_θ,), [k])

# #### Axial direction — standard B-splines

B_z = FunctionSpaces.create_bspline_space(0.0, H, n_z, p, k)

# #### Tensor-product space on the cylinder
#
# We pass the cylinder surface geometry explicitly to `TensorProductSpace`.
# The parametric geometry (a flat 2-D rectangle in ``(\theta,z)`` coordinates)
# is assembled from the parametric geometries of the two constituent spaces.

param_geo = Geometry.TensorProductGeometry((
    FunctionSpaces.get_parametric_geometry(GB_θ),
    FunctionSpaces.get_parametric_geometry(B_z),
))

B  = FunctionSpaces.TensorProductSpace((GB_θ, B_z), cylinder_geo, param_geo)
Λ⁰ = Forms.FormSpace(0, B, "ϕ")

# ### Forcing function and exact solution
#
# `AnalyticalFormField` evaluates `cylinder_geo` to obtain physical coordinates.
# The resulting matrix has shape ``(n_{\mathrm{pts}} \times 3)`` with columns
# ``(x, y, z) = (R\cos\theta,\, R\sin\theta,\, z)``.
# Because ``x/R = \cos\theta`` directly, no inverse-trigonometric function is needed.

λ = 1/R^2 + (π/H)^2   # Δ_g(cos θ · sin(πz/H)) = -λ · cos θ · sin(πz/H)

function forcing_function(x::Matrix{Float64})
    cosθ = x[:, 1] / R   # x = R cosθ  →  cosθ = x/R
    z    = x[:, 3]
    return [@. λ^2 * cosθ * sin(z)]   # H = π, so sin(πz/H) = sin(z)
end
f⁰ = Forms.AnalyticalFormField(0, forcing_function, cylinder_geo, "f⁰")

function exact_solution(x::Matrix{Float64})
    cosθ = x[:, 1] / R
    z    = x[:, 3]
    return [@. cosθ * sin(z)]
end

# ### Quadrature and weak form

canonical_qrule = Quadrature.tensor_product_rule((p + 1, p + 1), Quadrature.gauss_legendre)
dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(cylinder_geo))

# We reuse the dimension-agnostic biharmonic weak form from the [`Biharmonic`](@ref) example.

function zero_form_biharmonic(
    inputs::Assemblers.AbstractInputs, dΩ::Quadrature.AbstractGlobalQuadratureRule
)
    ψ⁰ = Assemblers.get_test_form(inputs)
    ϕ⁰ = Assemblers.get_trial_form(inputs)
    f⁰ = Assemblers.get_forcing(inputs)

    A = ∫(δ(d(ψ⁰)) ∧ ★(δ(d(ϕ⁰))), dΩ)
    lhs_expression = ((A,),)

    b = ∫(ψ⁰ ∧ ★(f⁰), dΩ)
    rhs_expression = ((b,),)

    return lhs_expression, rhs_expression
end

# ### Boundary conditions
#
# The true boundary of the cylinder consists of **only** the two circles at ``z = 0``
# and ``z = H``; the periodic angular direction carries no boundary.
# We therefore constrain only the degrees of freedom associated with those two circles.
#
# The 2-D dof_partition of a tensor-product space has ``3^2 = 9`` entries arranged in
# a ``3 \times 3`` grid (column-major, first index = ``\theta``, second = ``z``):
#
# | row ``\backslash`` col | ``z``-left | ``z``-interior | ``z``-right |
# |:---:|:---:|:---:|:---:|
# | ``\theta``-left     | 1 | 4 | 7 |
# | ``\theta``-interior | 2 | 5 | 8 |
# | ``\theta``-right    | 3 | 6 | 9 |
#
# Entries 1–3 cover **all** ``\theta``-partitions at ``z = 0``; entries 7–9 at ``z = H``.

function z_boundary_conditions(fe_space, value::Float64=0.0)
    dp = FunctionSpaces.get_dof_partition(fe_space)
    dofs = unique(vcat(dp[1][1], dp[1][2], dp[1][3],   # z = 0
                       dp[1][7], dp[1][8], dp[1][9]))   # z = H
    return Dict{Int, Float64}(i => value for i in dofs)
end

wfi = Assemblers.WeakFormInputs(Λ⁰, f⁰)
bc  = z_boundary_conditions(B)

# ### Assembly and solve

lhs_expressions, rhs_expressions = zero_form_biharmonic(wfi, dΩ)
weak_form = Assemblers.WeakForm(lhs_expressions, rhs_expressions, wfi)
A_mat, b_vec = Assemblers.assemble(weak_form, bc)
sol = vec(A_mat \ b_vec)
ϕ⁰  = Forms.build_form_field(Λ⁰, sol)

# ## Error analysis

ϕ⁰_exact = Forms.AnalyticalFormField(0, exact_solution, cylinder_geo, "ϕ_exact")

canonical_qrule_analysis = Quadrature.tensor_product_rule(
    (3 * p + 1, 3 * p + 1), Quadrature.gauss_legendre
)
dΩ_analysis = Quadrature.StandardQuadrature(
    canonical_qrule_analysis, Geometry.get_num_elements(cylinder_geo)
)

println("L² error (n = $n_θ): ", Analysis.L2_norm(ϕ⁰ - ϕ⁰_exact, dΩ_analysis))

# ### Convergence study
#
# We sweep over increasingly fine meshes and confirm the expected
# ``O(h^{p+1}) = O(h^4)`` convergence rate.

function build_cylinder(n, R, H)
    base   = Geometry.create_cartesian_box((0.0,), (2π,), (n,))
    ϕ_c(x::AbstractVector)   = [R * cos(x[1]), R * sin(x[1])]
    dϕ_c(x::AbstractVector)  = [-R * sin(x[1]); R * cos(x[1])]
    d²ϕ_c(x::AbstractVector) = (fill(-R * cos(x[1]), 1, 1), fill(-R * sin(x[1]), 1, 1))
    circ = Geometry.MappedGeometry(base, Geometry.Mapping((1, 2), ϕ_c, dϕ_c, d²ϕ_c))
    line = Geometry.create_cartesian_box((0.0,), (H,), (n,))
    return Geometry.TensorProductGeometry((circ, line))
end

function solve_biharmonic_cylinder(n, p, k, R, H)
    cyl   = build_cylinder(n, R, H)
    B_θ_  = FunctionSpaces.create_bspline_space(0.0, 2π, n, p, k)
    GB_θ_ = FunctionSpaces.GTBSplineSpace((B_θ_,), [k])
    B_z_  = FunctionSpaces.create_bspline_space(0.0, H, n, p, k)
    pg    = Geometry.TensorProductGeometry((
        FunctionSpaces.get_parametric_geometry(GB_θ_),
        FunctionSpaces.get_parametric_geometry(B_z_),
    ))
    B_    = FunctionSpaces.TensorProductSpace((GB_θ_, B_z_), cyl, pg)
    Λ⁰_   = Forms.FormSpace(0, B_, "ϕ")

    f⁰_   = Forms.AnalyticalFormField(0, forcing_function, cyl, "f⁰")
    qr    = Quadrature.tensor_product_rule((p + 1, p + 1), Quadrature.gauss_legendre)
    dΩ_   = Quadrature.StandardQuadrature(qr, Geometry.get_num_elements(cyl))
    wfi_  = Assemblers.WeakFormInputs(Λ⁰_, f⁰_)
    bc_   = z_boundary_conditions(B_)

    lhs_, rhs_ = zero_form_biharmonic(wfi_, dΩ_)
    wf_        = Assemblers.WeakForm(lhs_, rhs_, wfi_)
    A_, b_     = Assemblers.assemble(wf_, bc_)
    ϕ⁰_        = Forms.build_form_field(Λ⁰_, vec(A_ \ b_))

    ϕ_ex_ = Forms.AnalyticalFormField(0, exact_solution, cyl, "ϕ_exact")
    qr_an = Quadrature.tensor_product_rule((3 * p + 1, 3 * p + 1), Quadrature.gauss_legendre)
    dΩ_an = Quadrature.StandardQuadrature(qr_an, Geometry.get_num_elements(cyl))

    # Mantis.Plot.export_form_fields_to_vtk((ϕ⁰_, ϕ_ex_), "biharmonic_cylinder_n$(n)")
    # readline()

    return Analysis.L2_norm(ϕ⁰_ - ϕ_ex_, dΩ_an)
end

mesh_sizes = [8, 16, 32, 64]
h_vals     = 2π ./ mesh_sizes
errors     = [solve_biharmonic_cylinder(n, p, k, R, H) for n in mesh_sizes]

using GLMakie
using DisplayAs #hide

fig = Figure()
ax  = Axis(
    fig[1, 1];
    xlabel = "h",
    ylabel = "‖ϕₕ − ϕ‖_L²",
    title  = "Convergence on the cylinder surface",
    xscale = log10,
    yscale = log10,
)
scatterlines!(ax, h_vals, errors; label = "B-spline p=$p k=$k", markersize = 10)
C = errors[2] / h_vals[2]^(p + 1)
lines!(ax, h_vals, C .* h_vals .^ (p + 1); linestyle = :dash, label = "O(h^$(p+1))")
axislegend(ax; position = :rb)

fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide

# The error decreases at the expected rate ``O(h^4)``.
#
# ## Notes
#
# ### Why the boundary-condition helper is needed
#
# `Forms.set_dirichlet_boundary_conditions` constrains **all** boundary degrees of
# freedom, including those at the periodic join ``\theta = 0 = 2\pi``.  For the
# present solution ``\cos(\theta)\sin(z)``, which equals ``\sin(z) \neq 0`` at the
# join, those constraints would be incorrect and would degrade accuracy.
# The helper `z_boundary_conditions` targets only entries 1–3 and 7–9 of the
# dof_partition, which span all ``\theta``-partitions at ``z = 0`` and ``z = H``,
# leaving the periodic-join degrees of freedom free.
#
# ### Effect of the cylinder radius
#
# For ``R \neq 1`` the eigenvalue becomes ``\lambda = 1/R^2 + (\pi/H)^2`` and
# the forcing ``f = -\lambda^2\cos(\theta)\sin(\pi z/H)``.
# Because ``\cos\theta = x/R`` in physical coordinates, the same `forcing_function`
# and `exact_solution` work for any ``R`` as long as the global variable ``R`` is
# updated before constructing the geometry.
