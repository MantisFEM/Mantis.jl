# # Inspecting and visualizing a geometry

# Once you have built a geometry it is good practice to *look at it* before using it in a
# solve, both numerically (does it map where you expect? are its derivatives sensible?) and
# visually (in ParaView or Makie). This example covers the three evaluation routines every
# geometry supports, and the two ways to visualize one.

using Mantis

# We use a mildly curved 2D domain: a unit square deformed by a built-in curvilinear mapping,
# so that the Jacobian and Hessian are non-trivial.

square  = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (4, 4))
mapping = Geometry.create_curvilinear_mapping((0.0, 0.0), (1.0, 1.0))
geo     = Geometry.MappedGeometry(square, mapping)

println("manifold_dim = ", Geometry.get_manifold_dim(geo),
        ", image_dim = ", Geometry.get_image_dim(geo),
        ", num_elements = ", Geometry.get_num_elements(geo))

# ## Evaluate: where do points go?

# `evaluate` takes an `element_id` and points in canonical ``[0,1]^n`` coordinates, and
# returns a matrix with one row per point and one column per image dimension. Let us evaluate
# the four canonical corners of element 1:

corners = Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0]))
X = Geometry.evaluate(geo, 1, corners)
println("physical corners of element 1 (rows = points, cols = x,y):")
display(X)

# ## Jacobian: how is space stretched?

# `jacobian` returns, per point, the ``m \times n`` matrix ``\partial \Phi / \partial \xi``.
# Its determinant is the local volume-scaling factor; the absolute value, integrated, gives
# the element's measure. We look at the Jacobian at the centre of element 1:

centre = Points.CartesianPoints(([0.5], [0.5]))
J = Geometry.jacobian(geo, 1, centre)
println("\nJacobian of element 1 at its centre:")
display(J[1])
det_J = J[1][1, 1] * J[1][2, 2] - J[1][1, 2] * J[1][2, 1]  # 2×2 determinant
println("det(J) (local area scaling) = ", det_J)

# ## Hessian: curvature of the map

# `hessian` returns the second derivatives: for each point, one ``n \times n`` matrix per
# image component. For an undeformed Cartesian box these would all be zero; here they are not,
# which is exactly what makes the mapping curved:

H = Geometry.hessian(geo, 1, centre)
println("\nHessian blocks at the centre of element 1 (one per image component):")
display(H[1][1])
display(H[1][2])

# ## Visualizing in ParaView (VTK)

# The `Plot` module writes geometries to VTK files that high-order viewers such as
# [ParaView](https://www.paraview.org/) can render. Because it samples each element at a
# chosen polynomial `degree`, curved elements are shown faithfully. (We show the call as a
# code block rather than running it, so the example does not write files.)
#
# ```julia
# Mantis.Plot.plot(geo; vtk_filename = "my_geometry", degree = 3)
# # ... then open my_geometry.vtu in ParaView.
# ```

# ## Visualizing inline with Makie

# For a quick look without leaving Julia, evaluate the geometry along element boundaries and
# draw them. Constant-``\xi`` and constant-``\eta`` lines become the curved element edges of
# the physical domain.

using GLMakie
using DisplayAs #hide

fig = Figure(; size=(450, 450))
ax = Axis(fig[1, 1]; title="Mapped geometry", xlabel="x", ylabel="y", aspect=1)

nplot = 20
edge0 = Points.CartesianPoints((collect(LinRange(0.0, 1.0, nplot)), [0.0]))
edge1 = Points.CartesianPoints((collect(LinRange(0.0, 1.0, nplot)), [1.0]))
edge2 = Points.CartesianPoints(([0.0], collect(LinRange(0.0, 1.0, nplot))))
edge3 = Points.CartesianPoints(([1.0], collect(LinRange(0.0, 1.0, nplot))))
for e in 1:Geometry.get_num_elements(geo)
    for edge in (edge0, edge1, edge2, edge3)
        xy = Geometry.evaluate(geo, e, edge)
        lines!(ax, xy[:, 1], xy[:, 2]; color=:steelblue)
    end
end

fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide

# The next step is to put a function space on the geometry; see
# [B-spline spaces and basis functions](@ref).
