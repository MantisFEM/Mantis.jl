# # Tensor-product geometry
#
# This example demonstrates `TensorProductGeometry`, which assembles a higher-dimensional
# geometry by **tensoring** two or more lower-dimensional constituent geometries.
# The total manifold and image dimensions are the sums of the constituent dimensions,
# and elements are indexed as Cartesian products.
#
# We build two geometries step by step:
# 1. A **3D box** from a 2D Cartesian rectangle and a 1D line (the simplest case).
# 2. A **cylinder** from a mapped circle curve and a 1D line, where the constituent geometries
#    are themselves `MappedGeometry` objects.
#
# ### Background
#
# Given constituent geometries ``G_1, G_2`` with
# ``(\text{manifold\_dim}_i, \text{image\_dim}_i, n_i)``
# elements, the tensor product geometry has
#
# ```math
# \text{manifold\_dim} = \sum_i \text{manifold\_dim}_i,\quad
# \text{image\_dim}    = \sum_i \text{image\_dim}_i,\quad
# n = \prod_i n_i \text{ elements.}
# ```
#
# An element ``e`` of the tensor product is identified with a multi-index
# ``(e_1, e_2)`` (column-major ordering) and is evaluated at canonical points
# ``\xi = (\xi_1, \xi_2)`` as the concatenation of the constituent evaluations:
#
# ```math
# \phi_{e}(\xi) = \bigl(\phi^{(1)}_{e_1}(\xi_1),\; \phi^{(2)}_{e_2}(\xi_2)\bigr).
# ```
#
# The Jacobian is block-diagonal:
#
# ```math
# J_e(\xi) =
# \begin{pmatrix} J^{(1)}_{e_1}(\xi_1) & 0 \\ 0 & J^{(2)}_{e_2}(\xi_2) \end{pmatrix}.
# ```
#
# ### Part 1: 3D box from a 2D rectangle and a 1D line
#
# #### Implementation

using Mantis

# We create a 2D Cartesian rectangle and a 1D line.

rect = Geometry.create_cartesian_box((0.0, 0.0), (2.0, 1.0), (4, 3))
line = Geometry.create_cartesian_box((0.0,), (1.5,), (5,))

# Tensoring them yields a 3D box geometry.

box = Geometry.TensorProductGeometry((rect, line))

# The dimension and element counts are the expected sums and products.
Geometry.get_manifold_dim(box) == 2 + 1       # 3
Geometry.get_image_dim(box)    == 2 + 1       # 3
Geometry.get_num_elements(box) == 4 * 3 * 5  # 60

# #### Checking the construction
#
# At the first element's canonical origin ``\xi = (0,0,0)``, the physical point
# should be the corner ``(0, 0, 0)`` of the box.

p_origin = Points.CartesianPoints(([0.0], [0.0], [0.0]))
Geometry.evaluate(box, 1, p_origin)[1] ≈ 0.0
Geometry.evaluate(box, 1, p_origin)[2] ≈ 0.0
Geometry.evaluate(box, 1, p_origin)[3] ≈ 0.0

# The last element (global index ``4 \cdot 3 \cdot 5 = 60``) at ``\xi=(1,1,1)``
# should give the far corner ``(2, 1, 1.5)``.

p_corner = Points.CartesianPoints(([1.0], [1.0], [1.0]))
Geometry.evaluate(box, 60, p_corner)[1] ≈ 2.0
Geometry.evaluate(box, 60, p_corner)[2] ≈ 1.0
Geometry.evaluate(box, 60, p_corner)[3] ≈ 1.5

# The Jacobian at the first element's first corner is block-diagonal.
# The rectangle elements have size ``\Delta x = 0.5``, ``\Delta y = 1/3``;
# the line elements have size ``\Delta z = 0.3``.

J = Geometry.jacobian(box, 1, p_origin)
J[1] ≈ [0.5  0.0  0.0
        0.0  1/3  0.0
        0.0  0.0  0.3]

# #### Visualisation

using GLMakie
using DisplayAs #hide

fig = Figure(size=(420, 400))
ax = Axis3(fig[1, 1]; title="3D box", xlabel="x", ylabel="y", zlabel="z", aspect=:data)

# Draw the 12 edges of the box.
x1, x2 = 0.0, 2.0
y1, y2 = 0.0, 1.0
z1, z2 = 0.0, 1.5
edges = [
    ([x1, x2], [y1, y1], [z1, z1]),
    ([x1, x2], [y2, y2], [z1, z1]),
    ([x1, x2], [y1, y1], [z2, z2]),
    ([x1, x2], [y2, y2], [z2, z2]),
    ([x1, x1], [y1, y2], [z1, z1]),
    ([x2, x2], [y1, y2], [z1, z1]),
    ([x1, x1], [y1, y2], [z2, z2]),
    ([x2, x2], [y1, y2], [z2, z2]),
    ([x1, x1], [y1, y1], [z1, z2]),
    ([x2, x2], [y1, y1], [z1, z2]),
    ([x1, x1], [y2, y2], [z1, z2]),
    ([x2, x2], [y2, y2], [z1, z2]),
]
for (xs, ys, zs) in edges
    lines!(ax, xs, ys, zs; color=:steelblue, linewidth=2)
end

fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide

# ### Part 2: Cylinder from a mapped circle and a 1D line
#
# A more interesting example: the **lateral surface of a cylinder** arises naturally as
# the tensor product of a circle (a 1-dimensional manifold in 2D space) and a line segment.
# This yields a 2-dimensional manifold in 3D space, the same surface as in the
#  example, but constructed differently.
#
# #### Implementation
#
# First we build the circle as a `MappedGeometry`:
# a 1D Cartesian geometry on ``\theta \in [0,\, 2\pi]`` mapped by
# ``\phi_c(\theta) = (R\cos\theta,\, R\sin\theta)``.

R = 1.0
n_circle = 8   # 8 arc-elements
circle_base = Geometry.create_cartesian_box((0.0,), (2π,), (n_circle,))

function ϕ_circle(x::AbstractVector)
    θ = x[1]
    return [R * cos(θ), R * sin(θ)]
end
function dϕ_circle(x::AbstractVector)
    θ = x[1]
    return [-R * sin(θ); R * cos(θ)]   # 2-element vector → wrapped into SMatrix{2,1}
end

circle = Geometry.MappedGeometry(circle_base, Geometry.Mapping((1, 2), ϕ_circle, dϕ_circle))

# The circle is a 1D manifold embedded in 2D.
Geometry.get_manifold_dim(circle) == 1
Geometry.get_image_dim(circle)    == 2

# Now we create a vertical line segment and tensor-product it with the circle.

H = 2.0
n_height = 6
height_line = Geometry.create_cartesian_box((0.0,), (H,), (n_height,))

cylinder = Geometry.TensorProductGeometry((circle, height_line))

# The result is a 2-dimensional manifold (the cylinder surface) in 3D.
Geometry.get_manifold_dim(cylinder) == 2
Geometry.get_image_dim(cylinder)    == 3
Geometry.get_num_elements(cylinder) == n_circle * n_height   # 48

# #### Checking the construction
#
# The first global element corresponds to ``(e_{\rm circle}=1,\; e_{\rm height}=1)``.
# Its canonical origin ``\xi=(0,0)`` maps to the arc-element's start at ``\theta=0``,
# height ``z=0``, giving the physical point ``(R, 0, 0)``.

p_cyl = Points.CartesianPoints(([0.0], [0.0]))
Geometry.evaluate(cylinder, 1, p_cyl)[1] ≈ R
Geometry.evaluate(cylinder, 1, p_cyl)[2] ≈ 0.0
Geometry.evaluate(cylinder, 1, p_cyl)[3] ≈ 0.0

# The Jacobian at this point is block-diagonal.
# The circle arc-element has length ``\Delta\theta = 2\pi / n_{\rm circle}``
# and the height element has length ``\Delta z = H / n_{\rm height}``.
# At ``\theta=0``:
#
# ```math
# J\big|_{(0,0)} =
# \begin{pmatrix} -R\sin(0) & 0 \\ R\cos(0) & 0 \\ 0 & \Delta z \end{pmatrix}
# \begin{pmatrix} \Delta\theta & 0 \\ 0 & 1 \end{pmatrix}
# =
# \begin{pmatrix} 0 & 0 \\ R\,\Delta\theta & 0 \\ 0 & \Delta z \end{pmatrix}.
# ```

Δθ = 2π / n_circle
Δz = H / n_height
J_cyl = Geometry.jacobian(cylinder, 1, p_cyl)
J_cyl[1] ≈ [0.0 0.0; R * Δθ 0.0; 0.0 Δz]

# #### Visualisation
#
# We sample the cylinder surface by evaluating the mapping along its element boundaries
# and draw the arc-elements and height lines.

fig = Figure(size=(440, 480))
ax = Axis3(fig[1, 1]; title="Cylinder from TensorProductGeometry",
           xlabel="x", ylabel="y", zlabel="z", aspect=:data)

nplot = 40
θ_bps = LinRange(0.0, 2π, n_circle + 1)
z_bps = LinRange(0.0, H, n_height + 1)

# Horizontal rings at each height breakpoint.
for z in z_bps
    θs = LinRange(0.0, 2π, nplot)
    lines!(ax, R .* cos.(θs), R .* sin.(θs), fill(z, nplot); color=:steelblue)
end

# Vertical lines at each angular breakpoint.
for θ in θ_bps
    zs = LinRange(0.0, H, nplot)
    lines!(ax, fill(R * cos(θ), nplot), fill(R * sin(θ), nplot), zs; color=:steelblue)
end

fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide
