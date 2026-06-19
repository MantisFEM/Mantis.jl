# # Constructing Cartesian geometries

# Every problem in `Mantis` is posed on a *geometry*, a description of the physical domain
# together with its partition into elements. The simplest geometry is a **Cartesian box**: an
# axis-aligned interval, rectangle, or box with a tensor-product grid of elements. This example
# shows how to build them, how to read off their basic properties, and how to make a
# non-uniform (graded) mesh.

# ## The `(n, m)` picture

# A `Mantis` geometry maps a canonical reference element ``[0, 1]^n`` into physical space
# ``\mathbb{R}^m`` once per element. We call `n` the *manifold dimension* (the dimension you
# integrate over) and `m` the *image dimension* (the dimension of the space the domain lives
# in). For a plain box these are equal; they differ for embedded surfaces (see the
# [Tensor-product geometry](@ref) example, where a cylinder surface has `n = 2`, `m = 3`).

using Mantis

# ## A 1D line

# The main constructor is [`create_cartesian_box`](@ref Mantis.Geometry.create_cartesian_box),
# which takes the lower corner, the side lengths, and the number of elements per direction,
# each as a tuple. Here is the unit interval ``[0, 1]`` split into 4 elements:

line = Geometry.create_cartesian_box((0.0,), (1.0,), (4,))

println("line:  manifold_dim = ", Geometry.get_manifold_dim(line),
        ", image_dim = ", Geometry.get_image_dim(line),
        ", num_elements = ", Geometry.get_num_elements(line))

# To evaluate the geometry we hand `evaluate` an `element_id` and a set of points given in
# *canonical* coordinates (so ``0`` and ``1`` are the two ends of an element; see the
# [One-dimensional mapped geometry](@ref) example for why). The canonical endpoints `0` and
# `1` of element 1 are the physical points ``0`` and ``0.25``:

ends = Points.CartesianPoints(([0.0, 1.0],))
println("element 1 spans physical x = ", vec(Geometry.evaluate(line, 1, ends)))
println("element 4 spans physical x = ", vec(Geometry.evaluate(line, 4, ends)))

# ## A 2D rectangle and a 3D box

# Higher-dimensional boxes are created the same way; the number of elements is the product of
# the per-direction counts.

rectangle = Geometry.create_cartesian_box((0.0, 0.0), (2.0, 1.0), (3, 2))
box       = Geometry.create_cartesian_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2, 2, 2))

println("rectangle: ", Geometry.get_num_elements(rectangle), " elements (3 × 2)")
println("box:       ", Geometry.get_num_elements(box), " elements (2 × 2 × 2)")

# We can evaluate the rectangle at the canonical corner ``\xi = (0, 0)`` of its first element,
# which is the physical origin:

corner = Points.CartesianPoints(([0.0], [0.0]))
println("rectangle, element 1, ξ=(0,0) → ", vec(Geometry.evaluate(rectangle, 1, corner)))

# ## A non-uniform (graded) mesh

# `create_cartesian_box` always produces *uniform* elements. When you want element boundaries
# at specific, possibly unevenly-spaced, locations, build a
# [`CartesianGeometry`](@ref Mantis.Geometry.CartesianGeometry) directly from a tuple of
# *breakpoint* vectors, one strictly-increasing vector per direction. Here is a 1D mesh that
# is refined towards the left end:

breakpoints = [0.0, 0.05, 0.15, 0.35, 0.65, 1.0]
graded = Geometry.CartesianGeometry((breakpoints,))

element_sizes = [
    only(Geometry.evaluate(graded, e, Points.CartesianPoints(([1.0],)))) -
    only(Geometry.evaluate(graded, e, Points.CartesianPoints(([0.0],)))) for
    e in 1:Geometry.get_num_elements(graded)
]
println("graded mesh has ", Geometry.get_num_elements(graded), " elements of sizes ", element_sizes)

# The same idea extends to 2D/3D by passing one breakpoint vector per direction, e.g.
# `Geometry.CartesianGeometry((x_breakpoints, y_breakpoints))`.

# ## Where to go next

# A Cartesian geometry is the input to almost everything else: you build a B-spline space on
# it (see [B-spline spaces and basis functions](@ref)), wrap it in form spaces, and assemble.
# To curve or deform a
# Cartesian domain, see [One-dimensional mapped geometry](@ref) and
# [Two-dimensional mapped geometry](@ref); to combine geometries, see
# [Tensor-product geometry](@ref); and to look at a geometry, see
# [Inspecting and visualizing a geometry](@ref).
