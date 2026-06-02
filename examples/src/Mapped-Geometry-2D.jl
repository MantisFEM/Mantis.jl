# # Two-dimensional mapped geometry
#
# This example guides through creating a two-dimensional mapped geometry in *Mantis.jl*.
# As a concrete illustration we define a **quarter-annulus** by mapping a rectangle in
# polar coordinates ``(r, \theta)`` to the Cartesian plane.
#
# ### [Background](@id Mapped-Geometry-2D-Background)
#
# The construction follows the same two-ingredient recipe as the one-dimensional case
# (see [`One-dimensional mapped geometry`](@ref Mapped-Geometry-1D)):
# there is an original Cartesian geometry on the parametric rectangle
# ``(r, \theta) \in [r_0, r_1] \times [0, \pi/2]``,
# and a user-provided map
#
# ```math
# \phi(r, \theta) =
# \begin{pmatrix} r\cos\theta \\ r\sin\theta \end{pmatrix}.
# ```
#
# In two (and higher) dimensions, mapping functions receive a single parametric point
# ``\hat{x}`` as an `AbstractVector` — one entry per parametric dimension.
# The full chain rule
# ``D\phi/D\xi = (D\phi/D\hat{x})(D\hat{x}/D\xi)``
# is applied internally, so only derivatives with respect to the *physical* parametric
# coordinate ``\hat{x}`` are needed.
#
# The Jacobian ``D\phi`` is the ``2 \times 2`` matrix
#
# ```math
# D\phi(r, \theta) =
# \begin{pmatrix}
#   \cos\theta & -r\sin\theta \\
#   \sin\theta &  r\cos\theta
# \end{pmatrix},
# ```
#
# and the Hessian consists of one ``2\times 2`` matrix per image component:
#
# ```math
# H_x(r, \theta) =
# \begin{pmatrix} 0 & -\sin\theta \\ -\sin\theta & -r\cos\theta \end{pmatrix},
# \qquad
# H_y(r, \theta) =
# \begin{pmatrix} 0 & \cos\theta \\ \cos\theta & -r\sin\theta \end{pmatrix}.
# ```
#
# ### Implementation

using Mantis

# We set up the parametric (base) Cartesian geometry on the rectangle
# ``[r_0, r_1] \times [0, \pi/2]``.

r₀ = 1.0
r₁ = 2.0
num_elements = (3, 4)   # 3 elements in the r-direction, 4 in the θ-direction
parametric_geo = Geometry.create_cartesian_box((r₀, 0.0), (r₁ - r₀, π / 2), num_elements)

# Next we define the mapping ``\phi`` together with its first and second derivatives.
# Each function receives a single parametric point as an `AbstractVector`.

function ϕ_map(x::AbstractVector)
    r, θ = x[1], x[2]
    return [r * cos(θ), r * sin(θ)]
end
function dϕ_map(x::AbstractVector)
    r, θ = x[1], x[2]
    return [cos(θ)  -r * sin(θ)
            sin(θ)   r * cos(θ)]
end
function d²ϕ_map(x::AbstractVector)
    r, θ = x[1], x[2]
    return (
        [0.0  -sin(θ); -sin(θ)  -r * cos(θ)],  # Hessian for the x-component
        [0.0   cos(θ);  cos(θ)  -r * sin(θ)],  # Hessian for the y-component
    )
end

ϕ = Geometry.Mapping((2, 2), ϕ_map, dϕ_map, d²ϕ_map)

# We combine the parametric geometry and the mapping into a `MappedGeometry`.

geo = Geometry.MappedGeometry(parametric_geo, ϕ)

# ### Sanity checks
#
# The four corners of the parametric rectangle map as follows:
#
# | ``(r,\,\theta)``    | ``\phi(r,\,\theta)`` |
# |:-------------------:|:--------------------:|
# | ``(r_0,\;0)``       | ``(r_0,\;0)``        |
# | ``(r_1,\;0)``       | ``(r_1,\;0)``        |
# | ``(r_0,\;\pi/2)``   | ``(0,\;r_0)``        |
# | ``(r_1,\;\pi/2)``   | ``(0,\;r_1)``        |
#
# The first element covers ``r \in [r_0,\, r_0+\Delta r]``,
# ``\theta \in [0,\, \Delta\theta]``, so its canonical corner ``\xi=(0,0)``
# maps to the parametric point ``(r_0, 0)``.

nr, nθ = num_elements
Δr = (r₁ - r₀) / nr
Δθ = (π / 2) / nθ

p_origin = Points.CartesianPoints(([0.0], [0.0]))
p_corner = Points.CartesianPoints(([1.0], [1.0]))

# First element at ``\xi=(0,0)`` should give ``(r_0, 0)``.
Geometry.evaluate(geo, 1, p_origin)[1] ≈ r₀
Geometry.evaluate(geo, 1, p_origin)[2] ≈ 0.0

# Last element at ``\xi=(1,1)`` should give ``(0, r_1)``.
Geometry.evaluate(geo, nr * nθ, p_corner)[1] ≈ 0.0
Geometry.evaluate(geo, nr * nθ, p_corner)[2] ≈ r₁

# At the first element's first corner the chain rule gives (see [`Background`](@ref Mapped-Geometry-2D-Background)):
#
# ```math
# \frac{D\phi}{D\xi}\bigg|_{(r_0,\,0)} =
# D\phi(r_0,\,0)\,\mathrm{diag}(\Delta r,\,\Delta\theta) =
# \begin{pmatrix} \Delta r & 0 \\ 0 & r_0\,\Delta\theta \end{pmatrix}.
# ```

J = Geometry.jacobian(geo, 1, p_origin)
J[1] ≈ [Δr 0.0; 0.0 r₀ * Δθ]

# The full hessian at this corner is
# ``H_{\rm total}[i] = J_b^\top\, H_m[i]\, J_b`` with ``J_b = \mathrm{diag}(\Delta r, \Delta\theta)``:
#
# ```math
# H_x\big|_{(r_0,\,0)} = \begin{pmatrix} 0 & 0 \\ 0 & -r_0\,\Delta\theta^2 \end{pmatrix},
# \qquad
# H_y\big|_{(r_0,\,0)} = \begin{pmatrix} 0 & \Delta r\,\Delta\theta \\ \Delta r\,\Delta\theta & 0 \end{pmatrix}.
# ```

H = Geometry.hessian(geo, 1, p_origin)
H[1][1] ≈ [0.0 0.0; 0.0 -r₀ * Δθ^2]
H[1][2] ≈ [0.0 Δr * Δθ; Δr * Δθ 0.0]

# ### Visualisation
#
# We plot both domains side by side by evaluating the geometry along element boundary
# curves.  Constant-``r`` lines become circular arcs in the physical domain; constant-``\theta``
# lines remain straight radial segments.

using GLMakie
using DisplayAs #hide

fig = Figure(size=(700, 350))
ax_p = Axis(fig[1, 1]; title="Parametric domain", xlabel="r", ylabel="θ")
ax_q = Axis(fig[1, 2]; title="Physical domain",   xlabel="x", ylabel="y", aspect=1)

nplot = 60
r_bps = LinRange(r₀, r₁, nr + 1)
θ_bps = LinRange(0.0, π / 2, nθ + 1)

for r in r_bps
    lines!(ax_p, fill(r, nplot), LinRange(0.0, π / 2, nplot); color=:steelblue)
    θs = LinRange(0.0, π / 2, nplot)
    lines!(ax_q, r .* cos.(θs), r .* sin.(θs); color=:steelblue)
end
for θ in θ_bps
    lines!(ax_p, LinRange(r₀, r₁, nplot), fill(θ, nplot); color=:steelblue)
    rs = LinRange(r₀, r₁, nplot)
    lines!(ax_q, rs .* cos(θ), rs .* sin(θ); color=:steelblue)
end

fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide
