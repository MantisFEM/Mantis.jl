# # Function spaces: B-splines on lines, surfaces and volumes

# ## [Introduction](@id FSIntroduction)
# The [Geometry](Geometry.md) example built the domains on which `Mantis` computes. The
# next ingredient is a **function space**: a finite-dimensional space of functions on a
# geometry, spanned by a set of **basis functions**. Approximate solutions of PDEs are
# linear combinations of these basis functions. This example shows how function spaces are
# built and used, in one, two and three dimensions. We will:
#
# 1. build univariate **B-spline** spaces with uniform and non-uniform elements and with
#    different smoothness, and plot their basis functions;
# 2. query which basis functions are non-zero on which elements, and evaluate the basis
#    functions, their derivatives and their linear combinations;
# 3. build **tensor-product** spaces in 2D and 3D from univariate ones;
# 4. build spaces on the **mapped geometries** of the [Geometry](Geometry.md) example, and
#    compute derivatives with respect to the physical coordinates.
#
# The parametric domain, its elements, the canonical element and the mapping ``\Phi`` are
# explained in the [Geometry](Geometry.md) example, and we use them here without repeating
# their definitions. `Mantis` also provides multi-component, non-polynomial, periodic,
# polar and hierarchical spaces. They are built from the same ingredients and will be the
# subject of other examples.

# ## [Univariate B-spline spaces](@id FSUnivariate)
# ### [Piecewise polynomials and B-splines](@id FSUnivariateTheory)
# The [Heat Equation](HeatEquation.md) example, in its section *Choosing ``V_n``*,
# introduces the space ``V_n`` of piecewise polynomials of degree ``p`` on ``N`` elements
# that are ``C^k`` smooth at the breakpoints between elements, and shows that its
# dimension is ``n = (p+1)N - (k+1)(N-1)``. `Mantis` generalizes this space in one way:
# the smoothness can be chosen separately at each interior breakpoint. In the notation of
# [the Geometry example](@ref GeoParametric), with breakpoints
# ``\hat{x}_0 < \hat{x}_1 < \dots < \hat{x}_N``, a function of the space and its first
# ``k_i`` derivatives are continuous at ``\hat{x}_i``, for ``i = 1, \dots, N-1``. The
# **regularity** ``k_i`` satisfies ``-1 \leq k_i < p``, where ``k_i = -1`` means that the
# function may jump at ``\hat{x}_i``. Each interior breakpoint imposes ``k_i + 1``
# continuity conditions on the ``(p+1)N`` coefficients of the polynomials, so the
# dimension of the space is
# ```math
# n = (p+1)N - \sum_{i=1}^{N-1}(k_i + 1)\,,
# ```
# which is the formula of the Heat Equation example when all ``k_i`` are equal. `Mantis`
# stores one regularity per breakpoint, including the two end points. There it is always
# ``-1``: nothing is imposed beyond the ends of the domain.
#
# `Mantis` spans this space with **B-splines** ``\hat{B}_1, \dots, \hat{B}_n``. We will use
# the following properties of B-splines, and check most of them below:
# - **non-negativity**: ``\hat{B}_i(\hat{x}) \geq 0``;
# - **partition of unity**: ``\sum_{i=1}^n \hat{B}_i(\hat{x}) = 1`` for all ``\hat{x}``;
# - **local support**: each ``\hat{B}_i`` is non-zero on at most ``p+1`` consecutive
#   elements, and on each element exactly ``p+1`` B-splines are non-zero;
# - **end-point interpolation**: at each end point of the domain only one B-spline is
#   non-zero, and it equals 1. The same holds at a breakpoint with ``k_i = 0``.
#
# The last property was already used in the Heat Equation example to impose boundary
# conditions. On each element, every B-spline is a polynomial of degree ``p``, written in
# `Mantis` as a combination of ``p + 1`` polynomials defined once on the canonical element.
# How these combinations are stored is explained in
# [Extraction Coefficients](@ref ExtCoeffs-Implementation). We will not need the details
# here.
#
# We write ``\hat{B}_i`` with a hat because these functions are defined on the parametric
# domain. On a mapped geometry, they will be transported to the physical domain; see
# [Spaces on mapped geometries](@ref FSMapped).

# ### [Building univariate spaces](@id FSUnivariateCode)
# Besides `Mantis`, we use the `LinearAlgebra` standard library and, for the plots,
# `CairoMakie`.

using Mantis
using LinearAlgebra
using CairoMakie
using DisplayAs #hide

# A function space is built on top of a geometry, whose parametric domain provides the
# elements. We start from the interval ``[0, 2]`` with 4 elements, the
# [line](@ref GeoLine) of the Geometry example, and build the space of quadratic
# (``p = 2``), ``C^1`` (``k = 1``) B-splines on it. `FunctionSpaces.create_bspline_space`
# takes the geometry and, as tuples with one entry per direction, the degrees and the
# regularities at the interior breakpoints.

line = Geometry.create_cartesian_box((0.0,), (2.0,), (4,))
quadratic_space = FunctionSpaces.create_bspline_space(line, (2,), (1,))
FunctionSpaces.get_num_basis(quadratic_space)

# The dimension ``4 \cdot 3 - 3 \cdot 2 = 6`` agrees with the formula above. This is the
# space plotted in the Heat Equation example, there on ``[0, 1]``. The geometry is stored in
# the space:

FunctionSpaces.get_geometry(quadratic_space) === line

# The elements do not need to be uniform. On the
# [non-uniform line](@ref GeoLine) of the Geometry example, we build cubic, ``C^2``
# B-splines. The dimension is ``4 \cdot 4 - 3 \cdot 3 = 7``.

nonuniform_line = Geometry.CartesianGeometry(([0.0, 0.2, 0.6, 1.2, 2.0],))
cubic_space = FunctionSpaces.create_bspline_space(nonuniform_line, (3,), (2,))
FunctionSpaces.get_num_basis(cubic_space)

# To choose the regularity separately at each breakpoint, we use the
# `FunctionSpaces.BSplineSpace` constructor. It takes the geometry, the polynomials used on
# the canonical element (here the Bernstein polynomials of degree 3, see
# [Extraction Coefficients](@ref ExtCoeffs-Implementation)) and a vector with one
# regularity per breakpoint, including the end points. With the regularities
# ``(k_1, k_2, k_3) = (2, 1, 0)`` at ``\hat{x} = 0.5, 1, 1.5``, the dimension is
# ``4 \cdot 4 - (3 + 2 + 1) = 10``.

mixed_space = FunctionSpaces.BSplineSpace(
    line, FunctionSpaces.Bernstein(3), [-1, 2, 1, 0, -1]
)
FunctionSpaces.get_num_basis(mixed_space)

# Finally, if the geometry is not needed elsewhere, `FunctionSpaces.create_bspline_space`
# can also create it. It then takes the same first three arguments as
# `Geometry.create_cartesian_box`:

same_space = FunctionSpaces.create_bspline_space((0.0,), (2.0,), (4,), (2,), (1,))
FunctionSpaces.get_num_basis(same_space)

# ### [Plotting univariate spaces](@id FSUnivariatePlot)
# With `Makie` loaded, `Plot.plot_basis` draws all basis functions of a space. Since
# `Makie` also exports a name `Plot`, we write `Mantis.Plot` in full. The keyword arguments
# `label_prefix`, for the legend, and those of a `Makie` axis, such as `xlabel`, are
# optional. The black ticks mark the breakpoints. On the non-uniform line, the
# B-splines are narrower where the elements are shorter, and they are ``C^2``, so their
# graphs show no kinks:

fig = Mantis.Plot.plot_basis(
    cubic_space; label_prefix=L"\hat{B}_", xlabel=L"\hat{x}", ylabel=L"\hat{B}_i(\hat{x})"
)
fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide

# With mixed regularities, the smoothness drops from ``C^2`` at ``\hat{x} = 0.5`` to
# ``C^1`` at ``\hat{x} = 1`` and to ``C^0`` at ``\hat{x} = 1.5``. At the ``C^0``
# breakpoint, one B-spline equals 1 and has a kink, and all others vanish, as at the end
# points.

fig = Mantis.Plot.plot_basis(
    mixed_space; label_prefix=L"\hat{B}_", xlabel=L"\hat{x}", ylabel=L"\hat{B}_i(\hat{x})"
)
fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide

# ### [Basis functions, elements and supports](@id FSUnivariateQueries)
# The local support of the B-splines is what makes computations with them efficient: on
# each element, only ``p + 1`` basis functions need to be evaluated. `Mantis` gives this
# information in both directions. `FunctionSpaces.get_support` returns the elements on
# which a basis function is non-zero, and `FunctionSpaces.get_basis_indices` the basis
# functions that are non-zero on an element. For the quadratic space, ``\hat{B}_3`` is
# non-zero on elements 1 to 3, and on element 2 the non-zero basis functions are
# ``\hat{B}_2``, ``\hat{B}_3`` and ``\hat{B}_4``.

FunctionSpaces.get_support(quadratic_space, 3),
FunctionSpaces.get_basis_indices(quadratic_space, 2)

# These two relations are each other's inverse, and every element has ``p + 1 = 3`` basis
# functions:

all(1:FunctionSpaces.get_num_elements(quadratic_space)) do element_id
    basis_ids = FunctionSpaces.get_basis_indices(quadratic_space, element_id)
    return length(basis_ids) == 3 && all(
        element_id ∈ FunctionSpaces.get_support(quadratic_space, basis_id) for
        basis_id in basis_ids
    )
end

# `FunctionSpaces.get_dof_partition` splits the basis functions into those at the left
# end, those in the interior and those at the right end of each patch. Because of
# end-point interpolation, there is one basis function at each end. This partition is used
# to impose boundary conditions, see [Boundary Conditions](@ref FormsBCs). The keyword
# arguments `n_dofs_left` and `n_dofs_right` of `FunctionSpaces.create_bspline_space`
# change the number of basis functions at each end, for example to impose conditions on
# derivatives.

FunctionSpaces.get_dof_partition(quadratic_space)

# ### [Evaluating univariate spaces](@id FSUnivariateEvaluation)
# As for geometries, basis functions are evaluated element by element, at points of the
# canonical element. `FunctionSpaces.evaluate(space, element_id, ξ, num_derivatives)`
# returns two objects: the values and derivatives of the basis functions that are non-zero
# on the element, and their global indices. The values are nested vectors: `values[k+1]`
# contains the derivatives of order ``k``, `values[k+1][j]` the ``j``-th of these (in 1D
# there is only one derivative of each order) and `values[k+1][j][c]` its component ``c``
# (B-spline spaces have one component). The innermost object is a matrix with one row per
# point and one column per basis function. Here we evaluate the quadratic space and its
# first derivative at five points of element 2:

ξ_values = [0.0, 0.25, 0.5, 0.75, 1.0]
ξ = Points.TensorProductPoints((ξ_values,))
values, basis_ids = FunctionSpaces.evaluate(quadratic_space, 2, ξ, 1)
basis_ids

#-

values[1][1][1]

# Each row sums to one (partition of unity), so each row of the first derivative sums to
# zero:

sum_values = vec(sum(values[1][1][1]; dims=2))
sum_derivatives = vec(sum(values[2][1][1]; dims=2))
sum_values ≈ ones(5), isapprox(sum_derivatives, zeros(5); atol=1e-14)

# **What is new: derivatives with respect to the canonical coordinate.** As for the
# [Jacobian of a geometry](@ref GeoParametric), `Mantis` differentiates with respect to the
# canonical coordinate ``\xi``, not the parametric coordinate ``\hat{x}``. On element
# ``e``, ``\hat{x} = \hat{x}_{e-1} + h_e\,\xi``, so the chain rule gives
# ```math
# \frac{\mathrm{d}}{\mathrm{d}\xi}\hat{B}_i\big(\varphi_e(\xi)\big)
# = h_e\,\frac{\mathrm{d}\hat{B}_i}{\mathrm{d}\hat{x}}\,.
# ```
# This is why `evaluate` does not need the element lengths. On the interior elements of a
# uniform, quadratic, ``C^1`` space, the three non-zero B-splines are, as functions of
# ``\xi``,
# ```math
# \tfrac{1}{2}(1 - \xi)^2\,, \qquad \tfrac{1}{2}(1 + 2\xi - 2\xi^2)\,, \qquad
# \tfrac{1}{2}\xi^2\,,
# ```
# whatever the element length. Their derivatives with respect to ``\xi`` are
# ``-(1-\xi)``, ``1 - 2\xi`` and ``\xi``, and dividing by ``h_e = 1/2`` gives the
# derivatives with respect to ``\hat{x}``. On element 2:

x = ξ_values
values[1][1][1] ≈ hcat((1 .- x) .^ 2 ./ 2, (1 .+ 2x .- 2x .^ 2) ./ 2, x .^ 2 ./ 2),
values[2][1][1] ≈ hcat(-(1 .- x), 1 .- 2x, x)

# ### [Functions in a univariate space](@id FSUnivariateFunctions)
# A function of the space is a linear combination ``\hat{u} = \sum_i c_i \hat{B}_i``.
# Given the vector of all ``n`` coefficients ``c_i``,
# `FunctionSpaces.evaluate(space, element_id, ξ, num_derivatives, coefficients)` evaluates
# ``\hat{u}`` and its derivatives on an element. It returns the same nesting as before,
# with a vector of values per point instead of a matrix.
#
# B-splines reproduce polynomials of degree up to ``p``. For the linear function
# ``\hat{u}(\hat{x}) = \hat{x}`` the coefficients are known in closed form: they are the
# **Greville abscissae**, the averages of ``p`` consecutive knots, where the knots are the
# breakpoints, each repeated ``p - k_i`` times (``p + 1`` times at the end points).
# `FunctionSpaces.get_greville_points` returns them, as a tuple with one vector per
# direction. On the non-uniform cubic space, ``\hat{u}`` should be equal to ``\hat{x}``
# and, following the convention above, its derivative with respect to ``\xi`` should be
# equal to the element length ``h_e``:

(greville_points,) = FunctionSpaces.get_greville_points(cubic_space)
all(1:FunctionSpaces.get_num_elements(cubic_space)) do element_id
    û = FunctionSpaces.evaluate(cubic_space, element_id, ξ, 1, greville_points)
    x̂ = Geometry.evaluate(nonuniform_line, element_id, ξ)
    (h_e,) = Geometry.get_element_lengths(nonuniform_line, element_id)
    return û[1][1][1] ≈ x̂[:, 1] && û[2][1][1] ≈ fill(h_e, length(ξ_values))
end

# ## [Tensor-product spaces](@id FSTensorProduct)
# ### [Products of univariate B-splines](@id FSTensorProductTheory)
# On a tensor-product patch in ``n`` dimensions, see
# [the Geometry example](@ref GeoParametric), `Mantis` builds one univariate B-spline space
# per direction, each with its own degree ``p_k`` and regularity, and takes their products
# as basis functions:
# ```math
# \hat{B}_{\boldsymbol{i}}(\hat{\boldsymbol{x}})
# = \hat{B}^{1}_{i_1}(\hat{x}_1)\,\hat{B}^{2}_{i_2}(\hat{x}_2) \cdots
# \hat{B}^{n}_{i_n}(\hat{x}_n)\,, \qquad
# i = i_1 + (i_2 - 1)\,n_1 + (i_3 - 1)\,n_1 n_2 + \dots\,,
# ```
# where ``n_k`` is the dimension of the univariate space in direction ``k``. The basis
# functions are numbered with the first direction running fastest, with the same rule as
# the elements; see [Index Bookkeeping](@ref TensorProductsIndexing) and
# [Tensor-Product Function Spaces](@ref TensorProductsSpaces). The properties of the
# univariate B-splines carry over: the dimension is ``n_1 n_2 \cdots n_n``, the basis
# functions are non-negative and form a partition of unity, the support of
# ``\hat{B}_{\boldsymbol{i}}`` is the box of elements spanned by the supports of its
# factors, and ``(p_1 + 1) \cdots (p_n + 1)`` basis functions are non-zero on each element.
#
# **What is new: mixed derivatives.** In ``n`` dimensions there are ``n`` first
# derivatives ``\partial/\partial\xi_k``, ``n(n+1)/2`` second derivatives, and so on, all
# with respect to the canonical coordinates. `evaluate` stores all derivatives of order
# ``k`` in `values[k+1]`, in an order given by `Mantis.GeneralHelpers.get_derivative_idx`.
# This function takes the number of derivatives in each direction, for example `(1, 0)`
# for ``\partial/\partial\xi_1`` in 2D, and returns the position of that derivative.

# ### [2D: a square](@id FSSquare)
# On the [square](@ref GeoSquare) of the Geometry example, with ``4 \times 3`` elements, we
# build a space that is quadratic and ``C^1`` in the first direction, and linear and
# ``C^0`` in the second one. The univariate spaces have dimensions 6 and 4.

square = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (4, 3))
square_space = FunctionSpaces.create_bspline_space(square, (2, 1), (1, 0))
FunctionSpaces.get_num_basis(square_space),
FunctionSpaces.get_factor_num_basis(square_space)

# Following the numbering rule, basis function ``9 = 3 + (2 - 1) \cdot 6`` is the product
# ``\hat{B}^1_3\hat{B}^2_2``. Its first factor is non-zero on elements 1 to 3 in the first
# direction, and the second one on elements 1 and 2 in the second direction, so its
# support is made of the elements ``(i_1, i_2) \in \{1, 2, 3\} \times \{1, 2\}``, that is
# elements 1, 2, 3, 5, 6 and 7:

FunctionSpaces.get_factor_basis_ids(square_space, 9),
FunctionSpaces.get_support(square_space, 9)

# On element 6, ``(i_1, i_2) = (2, 2)``, the ``3 \times 2`` non-zero basis functions are
# the products of ``\hat{B}^1_2, \hat{B}^1_3, \hat{B}^1_4`` and ``\hat{B}^2_2,
# \hat{B}^2_3``:

FunctionSpaces.get_basis_indices(square_space, 6)

# The product structure can be checked directly. We evaluate the space on element 6 and
# its two univariate factor spaces on the corresponding factor elements. Since the points
# of `Points.TensorProductPoints` and the basis functions both have the first direction
# running fastest, the matrix of values of the product is the Kronecker product of the
# univariate matrices, in reverse order. The same holds for the first derivatives, where
# only the factor of the differentiated direction is differentiated.

space_1, space_2 = FunctionSpaces.get_factor_spaces(square_space)
ξ_1, ξ_2 = [0.0, 0.5, 1.0], [0.25, 0.75]
values, basis_ids = FunctionSpaces.evaluate(
    square_space, 6, Points.TensorProductPoints((ξ_1, ξ_2)), 1
)
values_1, _ = FunctionSpaces.evaluate(space_1, 2, Points.TensorProductPoints((ξ_1,)), 1)
values_2, _ = FunctionSpaces.evaluate(space_2, 2, Points.TensorProductPoints((ξ_2,)), 1)
d_1 = Mantis.GeneralHelpers.get_derivative_idx((1, 0))
d_2 = Mantis.GeneralHelpers.get_derivative_idx((0, 1))
values[1][1][1] ≈ kron(values_2[1][1][1], values_1[1][1][1]),
values[2][d_1][1] ≈ kron(values_2[1][1][1], values_1[2][1][1]),
values[2][d_2][1] ≈ kron(values_2[2][1][1], values_1[1][1][1])

# For second derivatives the order is less obvious, which is why it is best to always ask
# `get_derivative_idx` for the position:

[Mantis.GeneralHelpers.get_derivative_idx(key) for key in ((2, 0), (1, 1), (0, 2))]

# The partition into boundary and interior basis functions is also a tensor product. With
# (left end, interior, right end) in each direction and the first direction running
# fastest, there are 9 groups: the 4 corners (groups 1, 3, 7 and 9), the 4 edges (groups
# 2, 4, 6 and 8) and the interior (group 5), which contains
# ``(6 - 2)(4 - 2) = 8`` basis functions.

dof_partition = FunctionSpaces.get_dof_partition(square_space)[1]
length(dof_partition), dof_partition[5]

# For ``n = m = 2``, `Plot.plot_basis` draws the graphs of basis functions as surfaces;
# the keyword argument `ids` selects which ones. Here we draw ``\hat{B}_9``, the product
# of a quadratic and a linear B-spline, together with the elements. It is smooth in the
# first direction, while the linear factor gives it a ridge along ``\hat{x}_2 = 1/3``.

fig = Mantis.Plot.plot_basis(
    square_space;
    ids=9,
    draw_elements=true,
    xlabel=L"\hat{x}_1",
    ylabel=L"\hat{x}_2",
    zlabel=L"\hat{B}_9",
)
fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide

# ### [3D: a cube](@id FSCube)
# Nothing fundamentally new is needed in 3D. On the [cube](@ref GeoCube) of the Geometry
# example, with ``4 \times 3 \times 2`` elements, we build quadratic, ``C^1`` B-splines in
# all directions. The univariate spaces have dimensions 6, 5 and 4, so the space has
# dimension 120. On element 22, ``(i_1, i_2, i_3) = (2, 3, 2)``, ``3^3 = 27`` basis
# functions are non-zero.

cube = Geometry.create_cartesian_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (4, 3, 2))
cube_space = FunctionSpaces.create_bspline_space(cube, (2, 2, 2), (1, 1, 1))
FunctionSpaces.get_num_basis(cube_space), FunctionSpaces.get_num_basis(cube_space, 22)

# Basis function ``45 = 3 + (3 - 1) \cdot 6 + (2 - 1) \cdot 30`` is
# ``\hat{B}^1_3\hat{B}^2_3\hat{B}^3_2``. Each of its factors is non-zero on 3, 3 and 2
# elements, so it is non-zero on 18 elements. There are 3 first and 6 second derivatives:

ξ_3D = Points.TensorProductPoints(([0.0, 1.0], [0.0, 1.0], [0.0, 1.0]))
values, _ = FunctionSpaces.evaluate(cube_space, 22, ξ_3D, 2)
FunctionSpaces.get_factor_basis_ids(cube_space, 45),
length(FunctionSpaces.get_support(cube_space, 45)),
length.(values)

# `Plot.plot_basis` does not draw 3D spaces. Instead, we export the basis function to a
# VTK file and render it with ParaView, as for the geometries in the
# [Geometry](@ref GeoUnmappedPlot) example. The `Plot` module exports *forms*, which are
# the subject of a later example. For our purpose it is enough to know that a function is
# a ``0``-form: `Forms.FormSpace(0, space, label)` turns a function space into a space of
# ``0``-forms, and `Forms.FormField` combines it with a vector of coefficients; see
# [FormSpaces](@ref FormsSpaces) and [FormFields](@ref FormsFields). A basis function is the
# function whose coefficients are all zero, except for a 1 at its index. We write the
# files to a temporary folder; replace `output_dir` by any folder of your choice.

output_dir = mktempdir()

function export_basis_function(space, basis_id, filename; degree)
    coefficients = zeros(FunctionSpaces.get_num_basis(space))
    coefficients[basis_id] = 1.0
    basis_function = Forms.FormField(Forms.FormSpace(0, space, "B"), coefficients, "B")
    Mantis.Plot.export_form_fields_to_vtk(
        (basis_function,), filename; output_directory_tree=[output_dir], degree=degree
    )

    return nothing
end

export_basis_function(cube_space, 45, "cube_basis_45"; degree=4)

# The figure below shows ``\hat{B}_{45}`` on the half ``\hat{x}_2 \geq 1/2`` of the cube,
# together with the element edges of the whole cube. Its second factor is symmetric about
# ``\hat{x}_2 = 1/2``, so the cut face contains the maximum of the function. The function
# vanishes outside the ``3 \times 3 \times 2`` elements of its support.
#
# ![The basis function 45 on the half of the cube where the second coordinate is at least 1/2.](../assets/Examples/FunctionSpaces/function_spaces_cube.png)

# ## [Spaces on mapped geometries](@id FSMapped)
# ### [From the parametric domain to the physical domain](@id FSMappedTheory)
# So far the geometries were unmapped, and the basis functions were functions of the
# parametric coordinates ``\hat{\boldsymbol{x}}``. On a mapped geometry with
# [mapping](@ref GeoMapping) ``\Phi : \hat{\Omega} \to \mathcal{M}``, the basis functions
# are transported to the manifold ``\mathcal{M}`` by composition with the inverse of the
# mapping:
# ```math
# B_i(\boldsymbol{x}) = \hat{B}_i\big(\Phi^{-1}(\boldsymbol{x})\big)\,, \qquad
# \boldsymbol{x} \in \mathcal{M}\,.
# ```
# The space is thus first built on the parametric domain, exactly as before, and the
# mapping only changes where its functions live. As a consequence, the number of basis
# functions, their numbering, supports and partition, and their values at canonical
# points are the same on ``\mathcal{M}`` as on ``\hat{\Omega}``: `FunctionSpaces.evaluate`
# returns the derivatives of ``\hat{B}_i \circ \varphi_e = B_i \circ \Phi_e`` with respect
# to ``\boldsymbol{\xi}``, which do not depend on ``\Phi``.
#
# What changes are the derivatives with respect to the physical coordinates
# ``\boldsymbol{x}``. Write ``b = B_i \circ \Phi_e`` and ``\mathrm{J}_e`` for the
# [Jacobian](@ref GeoMapping) of ``\Phi_e``. Its columns are the tangent vectors
# ``\boldsymbol{t}_k = \partial\Phi_e/\partial\xi_k``, and the chain rule gives
# ``\partial b/\partial\xi_k = \nabla B_i \cdot \boldsymbol{t}_k``, that is
# ``\mathrm{J}_e^{\mathsf{T}}\,\nabla B_i = \nabla_{\boldsymbol{\xi}}\, b``.
# - For ``m = n``, the Jacobian is invertible, and
#   ``\nabla B_i = \mathrm{J}_e^{-\mathsf{T}}\,\nabla_{\boldsymbol{\xi}}\, b``.
# - For ``m > n``, the system has more unknowns than equations. The gradient of a function
#   on ``\mathcal{M}`` is a tangent vector, ``\nabla_{\mathcal{M}} B_i = \mathrm{J}_e
#   \boldsymbol{a}``, and substituting gives
#   ``\mathrm{J}_e^{\mathsf{T}}\mathrm{J}_e\,\boldsymbol{a} = \mathrm{g}\,\boldsymbol{a}
#   = \nabla_{\boldsymbol{\xi}}\, b``, with the [metric](@ref GeoMapping) ``\mathrm{g}``.
#   So ``\nabla_{\mathcal{M}} B_i = \mathrm{J}_e\,\mathrm{g}^{-1}\,
#   \nabla_{\boldsymbol{\xi}}\, b``, which reduces to the previous formula when
#   ``m = n``.
#
# `Mantis` stores the geometry in the space so that these transformations can be applied
# by the `Forms` module, which also handles quantities that transform differently from
# functions; see [How a `FormSpace` is evaluated](@ref FormsInternalEvaluateFormSpace).
# Here we apply the chain rule ourselves, to check the spaces we build. Later examples
# leave it to `Forms`.

# ### [Building a space on a mapped geometry](@id FSMappedCode)
# `FunctionSpaces.create_bspline_space(geometry, degrees, regularities)`, which we have
# used for Cartesian geometries, works unchanged for mapped ones: it takes the breakpoints
# from the parametric domain of the geometry, given by `Geometry.get_parametric_geometry`,
# and stores the geometry itself in the space. We use the geometry helpers introduced in
# the [Geometry](@ref GeoSummary) example. For the half annulus with ``4 \times 12``
# elements, quadratic, ``C^1`` B-splines give ``6 \cdot 14 = 84`` basis functions:

annulus = Geometry.create_annulus_sector((4, 12))
annulus_space = FunctionSpaces.create_bspline_space(annulus, (2, 2), (1, 1))
FunctionSpaces.get_num_basis(annulus_space)

# As explained above, the values and canonical derivatives are those of the same space on
# the parametric rectangle:

parametric_rectangle = Geometry.get_parametric_geometry(annulus)
rectangle_space = FunctionSpaces.create_bspline_space(parametric_rectangle, (2, 2), (1, 1))
ξ_2D = Points.TensorProductPoints(([0.0, 0.4, 1.0], [0.1, 0.9]))
all(
    FunctionSpaces.evaluate(annulus_space, e, ξ_2D, 1) ==
    FunctionSpaces.evaluate(rectangle_space, e, ξ_2D, 1) for e in 1:48
)

# **Lower-level constructors.** A space can also be assembled from univariate spaces with
# `FunctionSpaces.TensorProductSpace(factor_spaces, geometry, parametric_geometry)`, or
# from a mapping instead of a geometry, as in the [Biharmonic](Biharmonic.md) example. The
# univariate spaces must then have the same elements as the parametric domain, direction
# by direction and in the same order, and `Mantis` checks this. For example, the
# univariate spaces of the half annulus given in the wrong order are refused:

space_r = FunctionSpaces.create_bspline_space((1.0,), (1.0,), (4,), (2,), (1,))
space_θ = FunctionSpaces.create_bspline_space((0.0,), (1.0π,), (12,), (2,), (1,))
try
    FunctionSpaces.TensorProductSpace((space_θ, space_r), annulus, parametric_rectangle)
catch error
    println(error.msg)
end

# ### [2D: a half annulus](@id FSAnnulus)
# For ``n = m = 2``, `Plot.plot_basis` draws basis functions on the physical domain. Here
# we draw three basis functions with radial index 3 and angular indices 3, 7 and 12.
# Since ``B_i`` is non-zero on the same elements as ``\hat{B}_i``, each one is non-zero
# on a curved block of ``3 \times 3`` elements. The outer elements are larger, so the
# supports widen towards the outer circle.

fig = Mantis.Plot.plot_basis(
    annulus_space;
    ids=[3 + (3 - 1) * 6, 3 + (7 - 1) * 6, 3 + (12 - 1) * 6],
    draw_elements=true,
    xlabel=L"x_1",
    ylabel=L"x_2",
    zlabel=L"B_i",
)
fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide

# **Checking the half annulus.** We represent the function ``u(\boldsymbol{x}) = r =
# |\boldsymbol{x}|`` in the space. Since ``r`` is the first parametric coordinate,
# ``\hat{u}(r, \theta) = r`` is linear in ``r`` and constant in ``\theta``. Its
# coefficients are the [Greville abscissae](@ref FSUnivariateFunctions) in the radial
# direction, repeated for every angular index: by partition of unity in ``\theta``, the
# angular B-splines then sum to one. The exact gradient is the unit radial vector,
# ``\nabla u = \boldsymbol{x}/|\boldsymbol{x}|``, which we compare with
# ``\mathrm{J}_e^{-\mathsf{T}}\,\nabla_{\boldsymbol{\xi}}\,b`` on every element. As in the
# rest of `Mantis`, the coefficients are ordered with the first direction running
# fastest, which is the order in which Julia stores a matrix.

greville_r, greville_θ = FunctionSpaces.get_greville_points(annulus_space)
coefficients = vec([r for r in greville_r, θ in greville_θ])
all(1:Geometry.get_num_elements(annulus)) do element_id
    u = FunctionSpaces.evaluate(annulus_space, element_id, ξ_2D, 1, coefficients)
    x = Geometry.evaluate(annulus, element_id, ξ_2D)
    J = Geometry.jacobian(annulus, element_id, ξ_2D)
    return all(eachindex(J)) do i
        ∇ξ_u = [u[2][d_1][1][i], u[2][d_2][1][i]]
        ∇u = transpose(J[i]) \ ∇ξ_u
        return u[1][1][1][i] ≈ norm(x[i, :]) && ∇u ≈ x[i, :] / norm(x[i, :])
    end
end

# ### [2D in 3D: a helicoid](@id FSHelicoid)
# **What is new with respect to the half annulus: tangential derivatives.** On the
# [helicoid](@ref GeoHelicoid), ``m = 3 > n = 2``: the Jacobian is a ``3 \times 2``
# matrix, and the gradient must be computed with the metric. The space itself is built in
# the same way:

helicoid = Geometry.create_helicoid((4, 24))
helicoid_space = FunctionSpaces.create_bspline_space(helicoid, (2, 2), (1, 1))
FunctionSpaces.get_num_basis(helicoid_space)

# **Checking the helicoid.** We use the same function, ``u = r``, the distance to the
# vertical axis, ``r = \sqrt{x_1^2 + x_2^2}``. Its gradient in ``\mathbb{R}^3`` is
# ``(\cos\theta, \sin\theta, 0)``, which is the first column of ``\mathrm{D}\Phi``: the
# radial lines lie on the helicoid. So this gradient is tangent to the surface, and it
# is also the tangential gradient ``\nabla_{\mathcal{M}} u``. We compare it with
# ``\mathrm{J}_e\,\mathrm{g}^{-1}\,\nabla_{\boldsymbol{\xi}}\,b``, using
# `Geometry.metric` for ``\mathrm{g}``.

greville_r, greville_θ = FunctionSpaces.get_greville_points(helicoid_space)
coefficients = vec([r for r in greville_r, θ in greville_θ])
all(1:Geometry.get_num_elements(helicoid)) do element_id
    u = FunctionSpaces.evaluate(helicoid_space, element_id, ξ_2D, 1, coefficients)
    x = Geometry.evaluate(helicoid, element_id, ξ_2D)
    J = Geometry.jacobian(helicoid, element_id, ξ_2D)
    g, _ = Geometry.metric(helicoid, element_id, ξ_2D)
    return all(eachindex(J)) do i
        ∇ξ_u = [u[2][d_1][1][i], u[2][d_2][1][i]]
        ∇u = J[i] * (g[i] \ ∇ξ_u)
        r = norm(x[i, 1:2])
        return u[1][1][1][i] ≈ r && ∇u ≈ [x[i, 1] / r, x[i, 2] / r, 0.0]
    end
end

# **Visualizing the helicoid.** `Plot.plot_basis` cannot draw a function on a surface in
# ``\mathbb{R}^3``, so we export a basis function to VTK files, on the helicoid and on its
# parametric rectangle.

helicoid_rectangle = Geometry.get_parametric_geometry(helicoid)
helicoid_rectangle_space = FunctionSpaces.create_bspline_space(
    helicoid_rectangle, (2, 2), (1, 1)
)
basis_id = 3 + (5 - 1) * 6
export_basis_function(helicoid_space, basis_id, "helicoid_basis"; degree=4)
export_basis_function(
    helicoid_rectangle_space, basis_id, "helicoid_rectangle_basis"; degree=2
)

# The figure below shows the basis function with radial index 3 and angular index 5 on
# the parametric rectangle (left) and on the helicoid (right). The function is the same;
# only the domain is deformed.
#
# ![A basis function on the parametric rectangle (left) and on the helicoid (right).](../assets/Examples/FunctionSpaces/function_spaces_helicoid.png)

# ### [The spiral and the helical duct](@id FSOtherGeometries)
# The two remaining geometries of the Geometry example work in the same way. On the
# [spiral](@ref GeoSpiral), a curve in the plane, the formula of
# [the theory section](@ref FSMappedTheory) with ``n = 1`` and ``m = 2`` gives the
# derivative along the curve: ``\mathrm{g}`` is the squared length of the tangent vector
# ``\mathrm{J}_e``, so ``\nabla_{\mathcal{M}} B_i`` is the derivative with respect to
# ``\xi`` divided by ``\sqrt{\det\mathrm{g}}``, the length of the tangent vector, times
# the unit tangent.

spiral_space = FunctionSpaces.create_bspline_space(
    Geometry.create_archimedean_spiral(16), (2,), (1,)
)
duct_space = FunctionSpaces.create_bspline_space(
    Geometry.create_helical_duct((2, 36, 2)), (2, 2, 2), (1, 1, 1)
)
FunctionSpaces.get_num_basis(spiral_space), FunctionSpaces.get_num_basis(duct_space)

# ## [Summary and outlook](@id FSSummary)
# In this example we learned that:
# - a univariate B-spline space is defined by the breakpoints of a geometry, a degree
#   ``p`` and a regularity ``-1 \leq k_i < p`` at each interior breakpoint, and its
#   B-splines are non-negative, form a partition of unity, have local support and
#   interpolate at the end points;
# - `FunctionSpaces.get_support`, `FunctionSpaces.get_basis_indices` and
#   `FunctionSpaces.get_dof_partition` relate basis functions to elements and to the
#   boundary;
# - `FunctionSpaces.evaluate` evaluates the non-zero basis functions of an element, or a
#   linear combination of them, at canonical points. Derivatives are taken with respect to
#   the canonical coordinates, and in several dimensions their order is given by
#   `Mantis.GeneralHelpers.get_derivative_idx`;
# - in 2D and 3D, the basis functions are products of univariate B-splines, numbered with
#   the first direction running fastest;
# - on a mapped geometry, the basis functions are composed with ``\Phi^{-1}``. Their values
#   at canonical points do not change, and physical gradients follow from the Jacobian and
#   the metric of the geometry;
# - `Plot.plot_basis` draws the basis functions of spaces with ``n = m \leq 2``, and other
#   functions can be exported to VTK files as ``0``-forms.
#
# The functions used to build spaces are:
#
# | space | function |
# |:--- |:--- |
# | B-splines on a geometry | `FunctionSpaces.create_bspline_space(geometry, degrees, regularities)` |
# | B-splines on a box | `FunctionSpaces.create_bspline_space(starting_points, box_sizes, num_elements, degrees, regularities)` |
# | 1D, regularity per breakpoint | `FunctionSpaces.BSplineSpace(geometry, polynomials, regularities)` |
# | tensor product of given spaces | `FunctionSpaces.TensorProductSpace(factor_spaces, geometry, parametric_geometry)` |
#
# The spaces built here, on the geometries of the Geometry example, are the starting point
# for solving PDEs with differential forms in the following examples.
