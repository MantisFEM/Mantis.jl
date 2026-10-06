# # Geometry: curves, surfaces and volumes

# ## [Introduction](@id GeoIntroduction)
# Every computation in `Mantis` takes place on a *geometry*: function spaces, differential
# forms and integrals are all defined on one. This example shows how geometries are built
# and used, in one, two and three dimensions. We will:
#
# 1. build **unmapped** (Cartesian) geometries, a line, a square and a cube, with uniform
#    and non-uniform elements;
# 2. query the quantities attached to a geometry: element vertices and lengths, the
#    physical coordinates of points, the **Jacobian** and the **Hessian**;
# 3. introduce the **mapping** ``\Phi`` that turns an unmapped geometry into a curved one,
#    together with the **metric** it induces;
# 4. build four **mapped** geometries: an Archimedean spiral (a curve in the plane), a half
#    annulus (a curved planar domain), a helicoid (a surface in space) and a helical duct
#    (a curved volume);
# 5. export the geometries to VTK files and visualize them.
#
# The four mapped geometries are designed to be reused to solve PDEs in other examples:
# their mappings are smooth, their derivatives have closed-form expressions, and their
# lengths, areas and volumes are known exactly. We build each of them step by step, and
# then show the helper function of the `Geometry` module that creates it in one line.
# Other examples use these helpers and refer to this one for the explanation.

# ## [Unmapped geometries](@id GeoUnmapped)
# ### [Parametric domain, elements and the canonical element](@id GeoParametric)
# A single-patch geometry in `Mantis` starts from a **parametric domain**, a box
# ```math
# \hat{\Omega} = [a_1, b_1] \times \dots \times [a_n, b_n] \subset \mathbb{R}^n\,.
# ```
# Each direction ``k`` is split into ``N_k`` intervals by strictly increasing
# **breakpoints** ``a_k = \hat{x}_{k,0} < \hat{x}_{k,1} < \dots < \hat{x}_{k,N_k} = b_k``.
# The **elements** are the boxes obtained by choosing one interval in each direction:
# ```math
# \hat{\Omega}_e = [\hat{x}_{1,i_1-1}, \hat{x}_{1,i_1}] \times \dots \times
# [\hat{x}_{n,i_n-1}, \hat{x}_{n,i_n}]\,, \qquad
# e = i_1 + (i_2 - 1)\,N_1 + (i_3 - 1)\,N_1 N_2 + \dots\,,
# ```
# so that the first direction runs fastest in the element index ``e``. This is a
# **tensor product** of ``n`` one-dimensional partitions, and the numbering rule is the
# one used for every tensor-product object in `Mantis`, as described in
# [Index Bookkeeping](@ref TensorProductsIndexing). The box together with its elements is
# called a **patch**. A geometry can be made of several patches, but in this example we
# always use a single one.
#
# All evaluations in `Mantis` are done element by element, on the **canonical element**
# ``[0, 1]^n``, with coordinates ``\boldsymbol{\xi} = (\xi_1, \dots, \xi_n)``: as described
# in the [Points](@ref DocPointsModule) documentation, every point passed to an evaluation
# function is given in canonical coordinates. The **element map**
# ```math
# \varphi_e : [0, 1]^n \to \hat{\Omega}_e\,, \qquad
# \varphi_e(\boldsymbol{\xi}) = \begin{pmatrix} \hat{x}_{1,i_1-1} \\ \vdots \\
# \hat{x}_{n,i_n-1} \end{pmatrix} + \begin{pmatrix} h_{1,i_1} & & \\ & \ddots & \\
# & & h_{n,i_n} \end{pmatrix} \begin{pmatrix} \xi_1 \\ \vdots \\ \xi_n \end{pmatrix},
# ```
# with element lengths ``h_{k,i} = \hat{x}_{k,i} - \hat{x}_{k,i-1}``, takes a canonical
# point to the corresponding point of element ``e``. This is why most evaluation functions
# in `Mantis` take an element index *and* a set of canonical points: the first selects
# ``\varphi_e``, the second the points at which to evaluate it.
#
# An **unmapped** (or Cartesian) geometry is the parametric domain itself: element ``e`` is
# described by ``\varphi_e`` alone. Its **Jacobian** ``\mathrm{D}\varphi_e`` is the
# constant diagonal matrix of element lengths, its **Hessian** (its second derivatives) is
# zero, and the **measure** of the element (its length, area or volume) is the product of
# the element lengths. Note that the derivatives are taken with respect to the *canonical*
# coordinates ``\boldsymbol{\xi}``, which is why the element lengths appear in the
# Jacobian.

# ### [Building unmapped geometries](@id GeoUnmappedCode)
# Besides `Mantis`, we use the `LinearAlgebra` and `Printf` standard libraries.

using Mantis
using LinearAlgebra
using Printf

# #### [A line](@id GeoLine)
# The function `Geometry.create_cartesian_box` builds a patch with uniform elements. It
# takes three tuples, with one entry per direction: the starting points ``a_k``, the sizes
# ``b_k - a_k`` of the domain, and the numbers of elements ``N_k``. Here we build the
# interval ``[0, 2]`` with 4 elements. Note that the first two tuples must contain
# `Float64` values.

line = Geometry.create_cartesian_box((0.0,), (2.0,), (4,))
nothing #hide

# The dimensions of a geometry are part of its type. `Mantis` calls ``n``, the dimension of
# the parametric domain, the *manifold dimension*, and ``m``, the dimension of the space in
# which the geometry lives, the *image dimension*. For an unmapped geometry both are equal.

Geometry.get_manifold_dim(line), Geometry.get_image_dim(line)

# The number of patches and the number of elements are

Geometry.get_num_patches(line), Geometry.get_num_elements(line)

# For each element we can ask for its vertices (the breakpoints that bound it) and its
# lengths ``h_{k,i}``, one per direction.

Geometry.get_element_vertices(line, 2), Geometry.get_element_lengths(line, 2)

# To obtain non-uniform elements, we give the breakpoints directly to the
# `Geometry.CartesianGeometry` constructor, as a tuple with one vector of breakpoints per
# direction. Here the elements get longer towards the right end of ``[0, 2]``, as shown by
# their measures.

nonuniform_line = Geometry.CartesianGeometry(([0.0, 0.2, 0.6, 1.2, 2.0],))
[Geometry.get_element_measure(nonuniform_line, e) for e in 1:4]

# #### [A square](@id GeoSquare)
# In 2D every tuple has two entries. We build the unit square with 4 elements in the first
# direction and 3 in the second.

square = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (4, 3))
Geometry.get_manifold_dim(square), Geometry.get_image_dim(square)

# The number of elements is ``4 \times 3 = 12``. Following the numbering rule above,
# element 6 is ``(i_1, i_2) = (2, 2)``, so its vertices are
# ``[0.25, 0.5] \times [1/3, 2/3]``.

Geometry.get_num_elements(square), Geometry.get_element_vertices(square, 6)

# The figure below shows the square with the index of each element.
#
# ![The unit square with 4 × 3 elements and the element numbering.](../assets/Examples/Geometry/geometry_square.png)
#
# The directions are independent. In this non-uniform square the first direction is
# uniform and the second one is refined towards ``\hat{x}_2 = 0``:

nonuniform_square = Geometry.CartesianGeometry((
    [0.0, 0.25, 0.5, 0.75, 1.0], [0.0, 0.1, 0.3, 1.0]
))
Geometry.get_element_lengths(nonuniform_square, 6),
Geometry.get_element_measure(nonuniform_square, 6)

# #### [A cube](@id GeoCube)
# In 3D we build the unit cube with ``4 \times 3 \times 2`` elements. Element
# ``(i_1, i_2, i_3) = (2, 3, 2)`` has index ``2 + 2 \cdot 4 + 1 \cdot 12 = 22``.

cube = Geometry.create_cartesian_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (4, 3, 2))
Geometry.get_num_elements(cube), Geometry.get_element_vertices(cube, 22)

# ### [Evaluating unmapped geometries](@id GeoUnmappedEvaluation)
# To evaluate ``\varphi_e`` we first define the canonical points.
# `Points.TensorProductPoints` takes one vector of canonical coordinates per direction and
# forms all their combinations, with the first direction running fastest, as for the
# elements; see [TensorProductPoints](@ref PointsTensorProduct). With 2 coordinates in the
# first direction and 3 in the second we get 6 points:

ξ = Points.TensorProductPoints(([0.0, 1.0], [0.0, 0.5, 1.0]))
[point for point in ξ]

# `Geometry.evaluate` returns a matrix with one row per point and one column per image
# dimension. On element 6 of the non-uniform square, ``[0.25, 0.5] \times [0.1, 0.3]``, we
# get

Geometry.evaluate(nonuniform_square, 6, ξ)

# `Geometry.jacobian` returns one ``m \times n`` matrix per point. Here it is the diagonal
# matrix of element lengths ``h_{1,2} = 0.25`` and ``h_{2,2} = 0.2``, the same at every
# point, so we only show the first one:

Geometry.jacobian(nonuniform_square, 6, ξ)[1]

# `Geometry.hessian` returns, per point, a tuple with one ``n \times n`` matrix per image
# dimension, which contains the second derivatives of that component. For an unmapped
# geometry they are all zero.

Geometry.hessian(nonuniform_square, 6, ξ)[1]

# ### [Exporting and visualizing unmapped geometries](@id GeoUnmappedPlot)
# The `Plot` module does not draw anything itself. Instead, it writes VTK files, which
# can be opened with tools such as [ParaView](https://www.paraview.org). The function
# `Plot.export_geometry_to_vtk` samples each element at ``(p + 1)`` points per direction,
# with ``p`` given by the keyword argument `degree`, and stores it as a degree-``p``
# Lagrange cell. For ``n \geq 2`` it writes a second file, ending in `_wireframe`, with the
# edges of the elements. For unmapped geometries, ``p = 1`` is enough. We write the files
# to a temporary folder; replace `output_dir` by any folder of your choice.

output_dir = mktempdir()
for (name, geometry) in (
    ("line", line), ("nonuniform_line", nonuniform_line), ("square", square), ("cube", cube)
)
    Plot.export_geometry_to_vtk(
        geometry, name; output_directory_tree=[output_dir], degree=1
    )
end
readdir(output_dir)

# The figures below were rendered with ParaView from these files, coloring each element by
# its index. First, the uniform (top) and non-uniform (bottom) lines:
#
# ![The uniform (top) and non-uniform (bottom) lines, colored by element index.](../assets/Examples/Geometry/geometry_lines.png)
#
# and the cube. Its color increases slowly along the first direction and quickly along the
# last one, as follows from the numbering rule.
#
# ![The unit cube with 4 × 3 × 2 elements, colored by element index.](../assets/Examples/Geometry/geometry_cube.png)

# ## [Mapped geometries](@id GeoMapped)
# ### [The mapping ``\Phi``](@id GeoMapping)
# To describe curved geometries we add a second ingredient: a **mapping** that deforms the
# parametric domain. Let ``m \geq n``. A mapping is a function
# ```math
# \Phi : \hat{\Omega} \subset \mathbb{R}^n \to \mathbb{R}^m\,, \qquad
# \hat{\boldsymbol{x}} \mapsto \boldsymbol{x} = \Phi(\hat{\boldsymbol{x}})\,,
# ```
# that is smooth, injective, and whose Jacobian ``\mathrm{D}\Phi`` (an ``m \times n``
# matrix) has full rank ``n`` at every point. Under these conditions the image
# ``\mathcal{M} = \Phi(\hat{\Omega})`` is an ``n``-dimensional manifold in ``\mathbb{R}^m``,
# and ``\Phi`` is its parametrization: the parametric coordinates
# ``\hat{\boldsymbol{x}}`` label the points of ``\mathcal{M}``, and the full-rank condition
# makes sure that ``\mathcal{M}`` has a well-defined ``n``-dimensional tangent space
# everywhere. The manifold dimension of `Mantis` is ``n`` and the image dimension is
# ``m``. In this example we see
# - ``n = 1``, ``m = 2``: a curve in the plane (the spiral);
# - ``n = m = 2``: a deformed planar domain (the half annulus);
# - ``n = 2``, ``m = 3``: a surface in space (the helicoid);
# - ``n = m = 3``: a deformed volume (the helical duct).
#
# The mapping is shared by all elements. Composing it with the element maps gives the map
# from the canonical element to the physical element
# ``\mathcal{M}_e = \Phi(\hat{\Omega}_e)``:
# ```math
# \Phi_e := \Phi \circ \varphi_e : [0, 1]^n \to \mathcal{M}_e\,, \qquad
# \boldsymbol{x} = \Phi_e(\boldsymbol{\xi}) = \Phi\big(\varphi_e(\boldsymbol{\xi})\big)\,.
# ```
# The figure below shows these three domains for ``n = 2`` and ``m = 3``.
#
# ![The canonical element, the parametric domain and the manifold, with the maps between them.](../assets/Examples/Geometry/geometry_maps.svg)
#
# The [Geometry](@ref DocGeometryModule) documentation defines a geometry directly as the
# collection of the element maps ``\Phi_e`` (written ``\Phi_i`` there). A *mapped*
# geometry builds each of them from two pieces: the element maps ``\varphi_e``, which come
# from an unmapped geometry, and the mapping ``\Phi``, which is provided by the user. An
# unmapped geometry is the special case ``m = n`` and ``\Phi`` equal to the identity.
#
# **The Jacobian.** As for unmapped geometries, `Mantis` differentiates with respect to the
# canonical coordinates. By the chain rule,
# ```math
# \mathrm{J}_e(\boldsymbol{\xi}) := \mathrm{D}\Phi_e(\boldsymbol{\xi})
# = \mathrm{D}\Phi\big(\varphi_e(\boldsymbol{\xi})\big)\,\mathrm{D}\varphi_e\,,
# ```
# an ``m \times n`` matrix. The user only provides ``\mathrm{D}\Phi``, the derivatives with
# respect to the parametric coordinates, and `Mantis` applies the chain rule. The ``i``-th
# column of ``\mathrm{J}_e`` is the tangent vector ``\partial\Phi_e/\partial\xi_i`` to the
# coordinate line of ``\xi_i``.
#
# **The metric.** Lengths, areas and volumes on ``\mathcal{M}`` are measured with the
# **metric tensor**, the ``n \times n`` matrix of inner products of the tangent vectors,
# ```math
# g_{ij} = \frac{\partial\Phi_e}{\partial\xi_i}\cdot\frac{\partial\Phi_e}{\partial\xi_j}\,,
# \qquad \text{that is} \qquad
# \mathrm{g} = \mathrm{J}_e^{\mathsf{T}}\,\mathrm{J}_e\,.
# ```
# It is symmetric positive definite because ``\mathrm{J}_e`` has full rank. Its
# determinant gives the measure: the integral of a function ``f`` over element ``e`` is
# ```math
# \int_{\mathcal{M}_e} f \,\mathrm{d}\mathcal{M}
# = \int_{[0,1]^n} f\big(\Phi_e(\boldsymbol{\xi})\big)\,
# \sqrt{\det \mathrm{g}(\boldsymbol{\xi})}\,\mathrm{d}\boldsymbol{\xi}\,.
# ```
# When ``m = n`` the Jacobian is square and ``\sqrt{\det\mathrm{g}} = |\det\mathrm{J}_e|``,
# the familiar change-of-variables factor. When ``m > n`` the Jacobian has no determinant,
# and the metric is the only way to measure. This is why `Mantis` computes measures from
# the metric for every geometry.
#
# **The Hessian.** The second derivatives of ``\Phi_e`` form, for each of the ``m``
# components ``\Phi_{e,k}``, an ``n \times n`` matrix. Since ``\varphi_e`` is affine, its
# second derivatives vanish and the chain rule gives
# ```math
# \mathrm{H}_{e,k}(\boldsymbol{\xi}) := \mathrm{D}^2\Phi_{e,k}(\boldsymbol{\xi})
# = \mathrm{D}\varphi_e^{\mathsf{T}}\,
# \mathrm{D}^2\Phi_k\big(\varphi_e(\boldsymbol{\xi})\big)\,\mathrm{D}\varphi_e\,,
# \qquad k = 1, \dots, m\,.
# ```
# The Hessian gives the derivatives of the metric. These are needed by operators such as
# the [codifferential](@ref FormsCodifferential), which contains derivatives of the
# metric. Providing ``\mathrm{D}^2\Phi`` is therefore optional in `Mantis`, but required as
# soon as such operators are used.

# ### [Defining a mapping in `Mantis`](@id GeoMappingCode)
# The mapping and its derivatives are ordinary Julia functions of the parametric point
# ``\hat{\boldsymbol{x}}``, which they receive as a vector with ``n`` entries, even for
# ``n = 1``. They return:
# - ``\Phi``: a vector with ``m`` entries;
# - ``\mathrm{D}\Phi``: an ``m \times n`` matrix. For ``n = 1`` a vector with ``m``
#   entries is also accepted;
# - ``\mathrm{D}^2\Phi``: a tuple with one ``n \times n`` matrix per component of
#   ``\Phi``. For ``n = 1`` a tuple of numbers is also accepted.
#
# `Geometry.Mapping((n, m), Φ, DΦ, D²Φ)` collects the three functions, where the second
# derivatives can be left out when they are not needed. `Geometry.MappedGeometry` then
# combines a parametric domain, given as an unmapped geometry, with the mapping. The mapped
# geometry has the same elements as the parametric domain: the mapping changes where they
# are, not how many there are.
#
# For each geometry we will check the measure ``\sqrt{\det\mathrm{g}}`` at a few points,
# and that the total measure agrees with its exact value. The total measure is computed by integrating ``\sqrt{\det\mathrm{g}}`` over each
# canonical element with a tensor-product Gauss–Legendre rule, and summing over the
# elements. A quadrature rule in `Mantis` stores its nodes as canonical points, so they can
# be passed directly to `Geometry.metric`, which returns ``\mathrm{g}`` and
# ``\sqrt{\det\mathrm{g}}`` at each point.

function compute_measure(geometry, num_points_per_direction)
    n = Geometry.get_manifold_dim(geometry)
    quadrature_rule = Quadrature.tensor_product_rule(
        ntuple(_ -> num_points_per_direction, n), Quadrature.gauss_legendre
    )
    nodes = Quadrature.get_nodes(quadrature_rule)
    weights = Quadrature.get_weights(quadrature_rule)
    measure = 0.0
    for element_id in 1:Geometry.get_num_elements(geometry)
        _, sqrt_g = Geometry.metric(geometry, element_id, nodes)
        measure += sum(weights .* sqrt_g)
    end

    return measure
end
nothing #hide

# ### [1D: an Archimedean spiral](@id GeoSpiral)
# We map the parametric interval ``\hat{\Omega} = [0, 4\pi]`` onto two turns of an
# **Archimedean spiral**,
# ```math
# \Phi(\hat{x}) = r(\hat{x})\begin{pmatrix} \cos\hat{x} \\ \sin\hat{x} \end{pmatrix},
# \qquad r(\hat{x}) = r_0 + a\,\hat{x}\,,
# ```
# so that ``\hat{x}`` is the polar angle and the radius grows linearly with it. Successive
# turns are a distance ``2\pi a`` apart. We take ``r_0 = 1/2`` and ``a = 1/(4\pi)``: the
# radius grows from ``1/2`` to ``3/2``, and the turns are ``1/2`` apart. The derivatives
# are
# ```math
# \Phi'(\hat{x}) = \begin{pmatrix} a\cos\hat{x} - r\sin\hat{x} \\
# a\sin\hat{x} + r\cos\hat{x} \end{pmatrix}, \qquad
# \Phi''(\hat{x}) = \begin{pmatrix} -2a\sin\hat{x} - r\cos\hat{x} \\
# 2a\cos\hat{x} - r\sin\hat{x} \end{pmatrix}.
# ```
# The mapping satisfies the conditions above: it is injective because the radius strictly
# increases, and ``|\Phi'(\hat{x})| = \sqrt{a^2 + r^2} \geq a > 0``, so the Jacobian
# never vanishes.
#
# In 1D the quantities of the previous section are simple: ``\mathrm{J}_e`` is the tangent
# vector ``\Phi'(\hat{x})\,h_e``, the metric is its squared length, and
# ``\sqrt{\det\mathrm{g}} = |\Phi'(\hat{x})|\,h_e`` is the **speed** at which the curve is
# traversed, times the element length. The length of the spiral has a closed form, which
# we will use to verify the geometry. With the substitution ``u = r/a``,
# ```math
# L = \int_0^{4\pi}\sqrt{a^2 + r(\hat{x})^2}\,\mathrm{d}\hat{x}
# = \frac{a}{2}\Big[u\sqrt{1 + u^2} + \operatorname{arcsinh} u\Big]_{u_0}^{u_1}\,,
# \qquad u_0 = \frac{r_0}{a}\,,\quad u_1 = \frac{r_0}{a} + 4\pi\,.
# ```
#
# **Building the spiral.** The parametric domain is ``[0, 4\pi]`` with 16 uniform
# elements, and the mapping follows the conventions of
# [Defining a mapping in `Mantis`](@ref GeoMappingCode).

parametric_line = Geometry.create_cartesian_box((0.0,), (4π,), (16,))

r₀ = 0.5
a = 1 / (4π)

function spiral(x̂)
    r = r₀ + a * x̂[1]
    return [r * cos(x̂[1]), r * sin(x̂[1])]
end

function d_spiral(x̂)
    r = r₀ + a * x̂[1]
    return [a * cos(x̂[1]) - r * sin(x̂[1]), a * sin(x̂[1]) + r * cos(x̂[1])]
end

function dd_spiral(x̂)
    r = r₀ + a * x̂[1]
    return (-2a * sin(x̂[1]) - r * cos(x̂[1]), 2a * cos(x̂[1]) - r * sin(x̂[1]))
end

spiral_curve = Geometry.MappedGeometry(
    parametric_line, Geometry.Mapping((1, 2), spiral, d_spiral, dd_spiral)
)
Geometry.get_manifold_dim(spiral_curve), Geometry.get_image_dim(spiral_curve)

# **Checking the spiral.** The first point of the first element is ``\Phi(0) = (r_0, 0)``,
# and the last point of the last element is ``\Phi(4\pi) = (r_0 + 4\pi a, 0) = (3/2, 0)``.

first_point = Geometry.evaluate(spiral_curve, 1, Points.TensorProductPoints(([0.0],)))
last_point = Geometry.evaluate(spiral_curve, 16, Points.TensorProductPoints(([1.0],)))
first_point, last_point

# The measure ``\sqrt{\det\mathrm{g}}`` is the speed ``\sqrt{a^2 + r^2}`` times the element
# length. We check this at three canonical points of element 3. The parametric points are
# obtained by evaluating the parametric geometry, and `Geometry.get_element_lengths` gives
# ``h_e``.

ξ_1D = Points.TensorProductPoints(([0.0, 0.5, 1.0],))
e = 3
x̂ = Geometry.evaluate(parametric_line, e, ξ_1D)
(h_e,) = Geometry.get_element_lengths(parametric_line, e)
_, sqrt_g = Geometry.metric(spiral_curve, e, ξ_1D)
all(sqrt_g[i] ≈ sqrt(a^2 + (r₀ + a * x̂[i, 1])^2) * h_e for i in eachindex(sqrt_g))

# Finally, the length. The integrand ``\sqrt{a^2 + r^2}`` is not a polynomial, so the
# quadrature is not exact, but with five points per element the error is already at the
# level of round-off.

u₀ = r₀ / a
u₁ = r₀ / a + 4π
L_exact = a / 2 * ((u₁ * sqrt(1 + u₁^2) + asinh(u₁)) - (u₀ * sqrt(1 + u₀^2) + asinh(u₀)))
L = compute_measure(spiral_curve, 5)
@printf("length: %.12f, exact: %.12f, error: %.2e\n", L, L_exact, abs(L - L_exact))

# **The helper function.** `Geometry.create_archimedean_spiral(num_elements;
# inner_radius, radial_growth, angle)` creates this geometry. Its default parameters are
# the ones used here, so

spiral_helper = Geometry.create_archimedean_spiral(16)
all(
    Geometry.evaluate(spiral_helper, e, ξ_1D) ≈ Geometry.evaluate(spiral_curve, e, ξ_1D) for
    e in 1:16
)

# **Visualizing the spiral.** A curved element is not represented exactly by a degree-1
# cell, so we sample each element with a higher `degree`.

Plot.export_geometry_to_vtk(
    parametric_line, "parametric_line"; output_directory_tree=[output_dir], degree=1
)
Plot.export_geometry_to_vtk(
    spiral_curve, "spiral"; output_directory_tree=[output_dir], degree=6
)

# The figure below shows the parametric domain ``[0, 4\pi]`` (top) and the spiral
# (bottom), with matching colors for matching elements. All elements have the same length
# in the parametric domain, but on the spiral they get longer as the radius grows: this is
# the speed ``|\Phi'|`` at work.
#
# ![The parametric domain (top) and the Archimedean spiral (bottom), colored by element index.](../assets/Examples/Geometry/geometry_spiral.png)

# ### [2D: a half annulus](@id GeoAnnulus)
# The polar-coordinate map
# ```math
# \Phi(r, \theta) = \begin{pmatrix} r\cos\theta \\ r\sin\theta \end{pmatrix},
# \qquad (r, \theta) \in \hat{\Omega} = [1, 2] \times [0, \pi]\,,
# ```
# maps the rectangle ``\hat{\Omega}`` onto the upper half of the annulus between the
# circles of radii 1 and 2: a curved channel, or U-bend. Polar coordinates make
# closed-form solutions available on this domain, for example the harmonic function
# ``\ln r`` or the circular Couette flow. The Jacobian of the mapping and the Hessians of
# its two components are
# ```math
# \mathrm{D}\Phi = \begin{pmatrix} \cos\theta & -r\sin\theta \\
# \sin\theta & r\cos\theta \end{pmatrix}, \qquad
# \mathrm{D}^2\Phi_1 = \begin{pmatrix} 0 & -\sin\theta \\ -\sin\theta & -r\cos\theta
# \end{pmatrix}, \qquad
# \mathrm{D}^2\Phi_2 = \begin{pmatrix} 0 & \cos\theta \\ \cos\theta & -r\sin\theta
# \end{pmatrix}.
# ```
#
# **What is new with respect to the spiral: the determinant.** For ``n = m`` the Jacobian
# ``\mathrm{J}_e = \mathrm{D}\Phi\,\mathrm{D}\varphi_e`` is square, and the measure is
# ``\sqrt{\det\mathrm{g}} = |\det\mathrm{J}_e|``, here
# ```math
# \det\mathrm{J}_e = \det\mathrm{D}\Phi\,\det\mathrm{D}\varphi_e = r\,h_{1}h_{2} > 0\,,
# ```
# where ``h_1`` and ``h_2`` are the element lengths in ``r`` and ``\theta``. The sign of
# ``\det\mathrm{J}_e`` carries information that the metric does not: it is positive when
# ``\Phi_e`` preserves the orientation of the canonical element (the counter-clockwise
# order of its vertices) and negative when it reverses it. The full-rank condition makes
# sure that it never changes sign inside a patch. A mapping with a negative determinant,
# for example with ``\theta`` replaced by ``\pi - \theta``, describes the same domain with
# the opposite orientation, which flips the sign of integrals of differential forms. Here
# ``r \geq 1``, so ``\Phi`` preserves orientation. The area of the half annulus,
# ``\tfrac{\pi}{2}(2^2 - 1^2) = 3\pi/2``, will serve as a check.
#
# **Building the half annulus.** The parametric rectangle has 4 elements in ``r`` and 12
# in ``\theta``. Note that the tuples must contain `Float64` values, so we write `1.0π`
# instead of `π`, which in Julia is an irrational number. For ``n = 2`` the Jacobian is a
# ``2 \times 2`` matrix, and the second derivatives are a tuple of two ``2 \times 2``
# matrices.

parametric_rectangle = Geometry.create_cartesian_box((1.0, 0.0), (1.0, 1.0π), (4, 12))

function annulus(x̂)
    r, θ = x̂[1], x̂[2]
    return [r * cos(θ), r * sin(θ)]
end

function d_annulus(x̂)
    r, θ = x̂[1], x̂[2]
    return [
        cos(θ) -r*sin(θ)
        sin(θ) r*cos(θ)
    ]
end

function dd_annulus(x̂)
    r, θ = x̂[1], x̂[2]
    return (
        [
            0.0 -sin(θ)
            -sin(θ) -r*cos(θ)
        ],
        [
            0.0 cos(θ)
            cos(θ) -r*sin(θ)
        ],
    )
end

half_annulus = Geometry.MappedGeometry(
    parametric_rectangle, Geometry.Mapping((2, 2), annulus, d_annulus, dd_annulus)
)
nothing #hide

# **Checking the half annulus.** At the canonical points `ξ` defined
# [earlier](@ref GeoUnmappedEvaluation), ``\det\mathrm{J}_e`` should equal ``r\,h_1h_2``,
# which is positive, and ``\sqrt{\det\mathrm{g}}`` should equal it.

e = 7
x̂ = Geometry.evaluate(parametric_rectangle, e, ξ)
h₁, h₂ = Geometry.get_element_lengths(parametric_rectangle, e)
J = Geometry.jacobian(half_annulus, e, ξ)
_, sqrt_g = Geometry.metric(half_annulus, e, ξ)
all(det(J[i]) ≈ x̂[i, 1] * h₁ * h₂ ≈ sqrt_g[i] for i in eachindex(J))

# The area integrand ``r\,h_1h_2`` is linear in ``\xi_1`` and constant in ``\xi_2``, so
# two points per direction integrate it exactly, up to round-off.

A = compute_measure(half_annulus, 2)
A_exact = 3π / 2
@printf("area: %.12f, exact: %.12f, error: %.2e\n", A, A_exact, abs(A - A_exact))

# **The helper function.** `Geometry.create_annulus_sector(num_elements; radii, angle)`
# creates a sector of an annulus. Its default parameters give this half annulus.

annulus_helper = Geometry.create_annulus_sector((4, 12))
all(
    Geometry.evaluate(annulus_helper, e, ξ) ≈ Geometry.evaluate(half_annulus, e, ξ) for
    e in 1:48
)

# **Visualizing the half annulus.**

Plot.export_geometry_to_vtk(
    parametric_rectangle,
    "parametric_rectangle";
    output_directory_tree=[output_dir],
    degree=1,
)
Plot.export_geometry_to_vtk(
    half_annulus, "half_annulus"; output_directory_tree=[output_dir], degree=4
)

# The figure below shows the parametric rectangle (left) and the half annulus (right),
# with matching colors for matching elements. Lines of constant ``r`` become half circles
# and lines of constant ``\theta`` become radial segments. The elements near the outer
# circle are larger, because ``\det\mathrm{J}_e`` grows with ``r``.
#
# ![The parametric rectangle (left) and the half annulus (right), colored by element index.](../assets/Examples/Geometry/geometry_half_annulus.png)

# ### [2D: a helicoid](@id GeoHelicoid)
# Adding a third component that grows linearly with the angle lifts the annulus out of the
# plane, into a **helicoid**, the surface of a spiral ramp:
# ```math
# \Phi(r, \theta) = \begin{pmatrix} r\cos\theta \\ r\sin\theta \\ c\,\theta \end{pmatrix},
# \qquad (r, \theta) \in \hat{\Omega} = [1, 2] \times [0, 2\pi]\,.
# ```
# The ramp rises ``2\pi c`` per turn. We take ``c = 1/\pi``, so a full turn rises by 2.
# The first two components and their derivatives are those of the half annulus, and the
# third one adds a row to the Jacobian and a zero Hessian:
# ```math
# \mathrm{D}\Phi = \begin{pmatrix} \cos\theta & -r\sin\theta \\
# \sin\theta & r\cos\theta \\ 0 & c \end{pmatrix}, \qquad
# \mathrm{D}^2\Phi_3 = \begin{pmatrix} 0 & 0 \\ 0 & 0 \end{pmatrix}.
# ```
#
# **What is new with respect to the half annulus: a non-square Jacobian.** Now
# ``m = 3 > n = 2``: the Jacobian is a ``3 \times 2`` matrix, it has no determinant, and
# the measure must be computed from the metric. The two columns of ``\mathrm{D}\Phi`` are
# orthogonal, so the metric is diagonal:
# ```math
# \mathrm{D}\Phi^{\mathsf{T}}\mathrm{D}\Phi = \begin{pmatrix} 1 & 0 \\ 0 & r^2 + c^2
# \end{pmatrix}, \qquad
# \mathrm{g} = \mathrm{D}\varphi_e^{\mathsf{T}}\,\mathrm{D}\Phi^{\mathsf{T}}\mathrm{D}\Phi\,
# \mathrm{D}\varphi_e = \begin{pmatrix} h_1^2 & 0 \\ 0 & (r^2 + c^2)\,h_2^2 \end{pmatrix},
# \qquad \sqrt{\det\mathrm{g}} = \sqrt{r^2 + c^2}\,h_1h_2\,.
# ```
# Compared with the half annulus, the factor ``r`` became ``\sqrt{r^2 + c^2}``: the
# surface is stretched in the ``\theta`` direction because it also climbs. Its area is
# ```math
# A = \int_0^{2\pi}\!\!\int_1^2 \sqrt{r^2 + c^2}\,\mathrm{d}r\,\mathrm{d}\theta
# = 2\pi\left[\frac{r}{2}\sqrt{r^2 + c^2}
# + \frac{c^2}{2}\ln\!\Big(r + \sqrt{r^2 + c^2}\Big)\right]_1^2\,.
# ```
#
# The orientation, which for ``m = n`` was given by the sign of ``\det\mathrm{J}_e``, is
# given for a surface in space by the **unit normal**
# ``\boldsymbol{n} = (\boldsymbol{t}_1 \times \boldsymbol{t}_2)/
# |\boldsymbol{t}_1 \times \boldsymbol{t}_2|``, where ``\boldsymbol{t}_1`` and
# ``\boldsymbol{t}_2`` are the two columns of ``\mathrm{J}_e``. The full-rank condition
# makes sure that ``\boldsymbol{t}_1 \times \boldsymbol{t}_2`` never vanishes, so the
# normal is well defined everywhere. Its length is again the measure:
# ``|\boldsymbol{t}_1 \times \boldsymbol{t}_2| = \sqrt{\det\mathrm{g}}``.
#
# **Building the helicoid.** The parametric rectangle has 4 elements in ``r`` and 24 in
# ``\theta``. The mapping reuses the functions of the half annulus for its first two
# components.

parametric_rectangle_2π = Geometry.create_cartesian_box((1.0, 0.0), (1.0, 2π), (4, 24))

c = 1 / π

helicoid(x̂) = [annulus(x̂); c * x̂[2]]
d_helicoid(x̂) = [d_annulus(x̂); 0.0 c]
dd_helicoid(x̂) = (dd_annulus(x̂)..., zeros(2, 2))

helicoid_surface = Geometry.MappedGeometry(
    parametric_rectangle_2π, Geometry.Mapping((2, 3), helicoid, d_helicoid, dd_helicoid)
)
Geometry.get_manifold_dim(helicoid_surface), Geometry.get_image_dim(helicoid_surface)

# **Checking the helicoid.** The Jacobian is now a ``3 \times 2`` matrix,

e = 30
x̂ = Geometry.evaluate(parametric_rectangle_2π, e, ξ)
h₁, h₂ = Geometry.get_element_lengths(parametric_rectangle_2π, e)
J = Geometry.jacobian(helicoid_surface, e, ξ)
J[1]

# the metric is the diagonal matrix derived above, and the length of the cross product of
# the two columns of the Jacobian gives the same measure:

g, sqrt_g = Geometry.metric(helicoid_surface, e, ξ)
t₁ = [J[i][:, 1] for i in eachindex(J)]
t₂ = [J[i][:, 2] for i in eachindex(J)]
all(
    g[i] ≈ Diagonal([h₁^2, (x̂[i, 1]^2 + c^2) * h₂^2]) &&
        sqrt_g[i] ≈ sqrt(x̂[i, 1]^2 + c^2) * h₁ * h₂ for i in eachindex(g)
),
all(norm(t₁[i] × t₂[i]) ≈ sqrt_g[i] for i in eachindex(J))

# The area integrand is not a polynomial, so we use more quadrature points than for the
# half annulus.

A = compute_measure(helicoid_surface, 5)
F(r) = r / 2 * sqrt(r^2 + c^2) + c^2 / 2 * log(r + sqrt(r^2 + c^2))
A_exact = 2π * (F(2.0) - F(1.0))
@printf("area: %.12f, exact: %.12f, error: %.2e\n", A, A_exact, abs(A - A_exact))

# **The helper function.** `Geometry.create_helicoid(num_elements; radii, angle,
# rise_per_turn)` creates a helicoid. Its default parameters give this one.

helicoid_helper = Geometry.create_helicoid((4, 24))
all(
    Geometry.evaluate(helicoid_helper, e, ξ) ≈ Geometry.evaluate(helicoid_surface, e, ξ) for
    e in 1:96
)

# **Visualizing the helicoid.**

Plot.export_geometry_to_vtk(
    parametric_rectangle_2π,
    "parametric_rectangle_2pi";
    output_directory_tree=[output_dir],
    degree=1,
)
Plot.export_geometry_to_vtk(
    helicoid_surface, "helicoid"; output_directory_tree=[output_dir], degree=4
)

# The figure below shows the parametric rectangle (left) and the helicoid (right). The
# element colors increase with the angle ``\theta``, since the elements are numbered with
# the ``r`` direction running fastest.
#
# ![The parametric rectangle (left) and the helicoid (right), colored by element index.](../assets/Examples/Geometry/geometry_helicoid.png)

# ### [3D: a helical duct](@id GeoHelicalDuct)
# We thicken the helicoid in the vertical direction. A third parametric coordinate
# ``\zeta`` shifts the surface upwards:
# ```math
# \Phi(r, \theta, \zeta) = \begin{pmatrix} r\cos\theta \\ r\sin\theta \\
# \zeta + c\,\theta \end{pmatrix}, \qquad
# (r, \theta, \zeta) \in \hat{\Omega} = [1, 2] \times [0, 3\pi] \times [0, t]\,.
# ```
# The result is a duct with a rectangular cross-section of width 1 and height ``t``, that
# winds around the vertical axis for one and a half turns while rising ``2\pi c`` per
# turn: a curved 3D channel, for example for Stokes flow. Its bottom face ``\zeta = 0`` is
# the helicoid, extended to one and a half turns. We take again ``c = 1/\pi``, so the duct
# rises by 2 per turn, and ``t = 1``. The derivatives are
# ```math
# \mathrm{D}\Phi = \begin{pmatrix} \cos\theta & -r\sin\theta & 0 \\
# \sin\theta & r\cos\theta & 0 \\ 0 & c & 1 \end{pmatrix},
# ```
# and the Hessians of the first two components are those of the half annulus, padded with
# zeros for ``\zeta``. The Hessian of the third component is zero.
#
# Expanding the determinant along the last column gives ``\det\mathrm{D}\Phi = r``. The
# climb ``c\,\theta`` shears the duct vertically, and a shear does not change volumes.
# The volume of the duct is therefore that of a flat annular sector of the same
# thickness,
# ```math
# V = \int_0^{t}\!\!\int_0^{3\pi}\!\!\int_1^2 r\,\mathrm{d}r\,\mathrm{d}\theta
# \,\mathrm{d}\zeta = \frac{3}{2}\cdot 3\pi\cdot t = \frac{9\pi}{2}\,.
# ```
#
# **What is new: local and global invertibility.** The conditions on ``\Phi`` stated in
# [The mapping ``\Phi``](@ref GeoMapping) are of two kinds. The full-rank condition, here
# ``\det\mathrm{D}\Phi \neq 0``, is local: it guarantees that ``\Phi`` is invertible near
# each point. Injectivity is global: two distant points of ``\hat{\Omega}`` must not be
# mapped to the same physical point. The helical duct shows that the two are different.
# Its determinant is ``r > 0`` for *any* ``c`` and ``t``. However, a point at angle
# ``\theta`` and one at ``\theta + 2\pi`` lie on the same vertical line, a height
# ``2\pi c`` apart. If the duct is thicker than that, ``t \geq 2\pi c``, consecutive turns
# overlap and ``\Phi`` is not injective, even though the Jacobian is invertible
# everywhere. With our values, ``t = 1 < 2\pi c = 2``, so the turns are separated by a gap
# of height 1. The same happens in 2D: a polar-coordinate map with ``\theta`` ranging over
# more than ``2\pi`` has a positive determinant but covers part of the annulus twice.
# `Mantis` cannot detect a non-injective mapping in general. It is the user's
# responsibility to provide one that is injective.
#
# **Building the helical duct.** The parametric box has 2 elements in ``r``, 36 in
# ``\theta`` and 2 in ``\zeta``. For ``n = 3`` the Jacobian is a ``3 \times 3`` matrix
# and the second derivatives are a tuple of three ``3 \times 3`` matrices.

t = 1.0
parametric_box = Geometry.create_cartesian_box((1.0, 0.0, 0.0), (1.0, 3π, t), (2, 36, 2))

function helical_duct(x̂)
    r, θ, ζ = x̂[1], x̂[2], x̂[3]
    return [r * cos(θ), r * sin(θ), ζ + c * θ]
end

function d_helical_duct(x̂)
    r, θ = x̂[1], x̂[2]
    return [
        cos(θ) -r*sin(θ) 0.0
        sin(θ) r*cos(θ) 0.0
        0.0 c 1.0
    ]
end

function dd_helical_duct(x̂)
    r, θ = x̂[1], x̂[2]
    return (
        [
            0.0 -sin(θ) 0.0
            -sin(θ) -r*cos(θ) 0.0
            0.0 0.0 0.0
        ],
        [
            0.0 cos(θ) 0.0
            cos(θ) -r*sin(θ) 0.0
            0.0 0.0 0.0
        ],
        zeros(3, 3),
    )
end

duct = Geometry.MappedGeometry(
    parametric_box, Geometry.Mapping((3, 3), helical_duct, d_helical_duct, dd_helical_duct)
)
nothing #hide

# **Checking the helical duct.** A simple test of the local condition is to evaluate
# ``\det\mathrm{J}_e`` at the quadrature points of every element and check that it never
# changes sign. This is worth doing for any new mapping, because a sign error in
# ``\mathrm{D}\Phi`` or an invalid choice of parameters is caught immediately. For the
# duct, ``\det\mathrm{J}_e = r\,h_1h_2h_3`` should be positive everywhere.

nodes = Quadrature.get_nodes(
    Quadrature.tensor_product_rule((2, 2, 2), Quadrature.gauss_legendre)
)
min_det = minimum(1:Geometry.get_num_elements(duct)) do element_id
    return minimum(det, Geometry.jacobian(duct, element_id, nodes))
end
min_det > 0

# The volume integrand ``r\,h_1h_2h_3`` is linear in ``\xi_1`` and constant in the other
# directions, so two points per direction integrate it exactly, up to round-off.

V = compute_measure(duct, 2)
V_exact = 9π / 2
@printf("volume: %.12f, exact: %.12f, error: %.2e\n", V, V_exact, abs(V - V_exact))

# **The helper function.** `Geometry.create_helical_duct(num_elements; radii, angle,
# thickness, rise_per_turn)` creates a helical duct. Its default parameters give this one.

duct_helper = Geometry.create_helical_duct((2, 36, 2))
all(
    Geometry.evaluate(duct_helper, e, nodes) ≈ Geometry.evaluate(duct, e, nodes) for
    e in 1:144
)

# Unlike our hand-written mapping, the helper checks the global condition: it refuses a
# duct whose turns would overlap.

try
    Geometry.create_helical_duct((2, 36, 2); thickness=2.5)
catch error
    println(error.msg)
end

# **Visualizing the helical duct.**

Plot.export_geometry_to_vtk(
    parametric_box, "parametric_box"; output_directory_tree=[output_dir], degree=1
)
Plot.export_geometry_to_vtk(
    duct, "helical_duct"; output_directory_tree=[output_dir], degree=4
)

# The figure below shows the parametric box (left) and the helical duct (right), colored
# by element index. The gap between consecutive turns is the margin ``2\pi c - t = 1``
# that keeps the mapping injective.
#
# ![The parametric box (left) and the helical duct (right), colored by element index.](../assets/Examples/Geometry/geometry_helical_duct.png)

# ## [Summary and outlook](@id GeoSummary)
# In this example we learned that:
# - an unmapped geometry is a tensor-product patch of elements defined by its breakpoints,
#   created with `Geometry.create_cartesian_box` (uniform elements) or
#   `Geometry.CartesianGeometry` (any breakpoints). Its elements, as well as the points of
#   `Points.TensorProductPoints`, are numbered with the first direction running fastest;
# - all evaluations are done on the canonical element, and element ``e`` is reached with
#   the element map ``\varphi_e``;
# - a mapped geometry composes the element maps with a user-provided mapping
#   ``\Phi: \hat{\Omega}\subset\mathbb{R}^n \to \mathbb{R}^m``, built with
#   `Geometry.Mapping` and `Geometry.MappedGeometry`;
# - `Geometry.jacobian` and `Geometry.hessian` differentiate with respect to the canonical
#   coordinates, and `Geometry.metric` gives the metric and the measure
#   ``\sqrt{\det\mathrm{g}}``. For ``m = n`` the measure is ``|\det\mathrm{J}_e|`` and its
#   sign gives the orientation; for ``m > n`` only the metric can measure;
# - the Jacobian must have full rank everywhere (a local condition, easy to check), and the
#   mapping must be injective (a global condition, the user's responsibility);
# - `Plot.export_geometry_to_vtk` writes a geometry to VTK files for visualization.
#
# The four mapped geometries are available as helper functions, which other examples use
# directly:
#
# | geometry | ``(n, m)`` | helper |
# |:--- |:---: |:--- |
# | Archimedean spiral | ``(1, 2)`` | `Geometry.create_archimedean_spiral` |
# | annulus sector | ``(2, 2)`` | `Geometry.create_annulus_sector` |
# | helicoid | ``(2, 3)`` | `Geometry.create_helicoid` |
# | helical duct | ``(3, 3)`` | `Geometry.create_helical_duct` |
#
# For another 1D mapped geometry, one that maps an interval onto another interval
# (``n = m = 1``), see the [one-dimensional mapped geometry](Mapped-Geometry-1D.md)
# example.
