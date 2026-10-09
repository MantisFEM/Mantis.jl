# # Differential forms: construction, projection and operations

# ## [Introduction](@id DFIntroduction)
# `Mantis` represents every field as a **differential form**: a function is a ``0``-form,
# a quantity that is integrated along curves (such as a circulation) is a ``1``-form, a
# quantity integrated over surfaces (such as a flux) is an ``(n-1)``-form, and a density
# integrated over volumes is an ``n``-form. Why this language is used, and how it relates
# to the familiar ``\nabla``, ``\nabla\times`` and ``\nabla\cdot``, is explained on the
# [differential forms theory page](@ref TheoryForms). The `Forms` module that implements
# them is described on the [Forms](@ref) page.
#
# This example shows how to use the `Forms` module, building on the geometries of the
# [Geometry](Geometry.md) example and the B-spline spaces of the
# [Function Spaces](FunctionSpaces.md) example. We will:
#
# 1. construct forms of every rank in 1D, 2D and 3D from function spaces and coefficients,
#    and from analytical expressions;
# 2. export forms to VTK files to visualize them;
# 3. compute the ``L^2`` projection of analytical forms onto finite element forms, on
#    coarse and fine meshes and on a curved geometry;
# 4. apply the exterior derivative ``\mathrm{d}``, the Hodge star ``\star`` and the wedge
#    product ``\wedge`` to finite element forms, and compare the results with the exact
#    ones.

# ## [Constructing forms](@id DFConstruction)
# ### [Form spaces and form fields](@id DFSpacesAndFields)
# On a geometry of manifold dimension ``n``, with the parametric coordinates
# ``\hat{\boldsymbol{x}} = (\hat{x}_1, \dots, \hat{x}_n)`` of the
# [Geometry](@ref GeoParametric) example, a ``k``-form has ``\binom{n}{k}`` components,
# one per basis ``k``-form
# ``\mathrm{d}\hat{x}_{i_1}\wedge\dots\wedge\mathrm{d}\hat{x}_{i_k}``. We write
# ``\omega^k_c`` for the ``c``-th basis ``k``-form, in the order used by `Mantis`:
#
# | ``n`` | ``1``-forms | ``2``-forms | ``3``-forms |
# |:---: |:--- |:--- |:--- |
# | 1 | ``\mathrm{d}\hat{x}_1`` | | |
# | 2 | ``\mathrm{d}\hat{x}_1``, ``\mathrm{d}\hat{x}_2`` | ``\mathrm{d}\hat{x}_1\wedge\mathrm{d}\hat{x}_2`` | |
# | 3 | ``\mathrm{d}\hat{x}_1``, ``\mathrm{d}\hat{x}_2``, ``\mathrm{d}\hat{x}_3`` | ``\mathrm{d}\hat{x}_2\wedge\mathrm{d}\hat{x}_3``, ``\mathrm{d}\hat{x}_3\wedge\mathrm{d}\hat{x}_1``, ``\mathrm{d}\hat{x}_1\wedge\mathrm{d}\hat{x}_2`` | ``\mathrm{d}\hat{x}_1\wedge\mathrm{d}\hat{x}_2\wedge\mathrm{d}\hat{x}_3`` |
#
# For example, a ``1``-form in 2D and a ``2``-form in 3D are
# ```math
# \alpha^1 = a_1\,\mathrm{d}\hat{x}_1 + a_2\,\mathrm{d}\hat{x}_2\,, \qquad
# \alpha^2 = a_1\,\mathrm{d}\hat{x}_2\wedge\mathrm{d}\hat{x}_3
# + a_2\,\mathrm{d}\hat{x}_3\wedge\mathrm{d}\hat{x}_1
# + a_3\,\mathrm{d}\hat{x}_1\wedge\mathrm{d}\hat{x}_2\,.
# ```
#
# In `Mantis`, a finite element form is built in two steps.
#
# **Form spaces.** `Forms.FormSpace(k, space, label)` combines a function space with the
# basis ``k``-forms. The function space must have one component per component of the
# form, that is, for each ``c = 1, \dots, \binom{n}{k}``, a set of scalar basis functions
# ``\hat{B}_{c,1}, \dots, \hat{B}_{c,M_c}``, such as the B-splines of the
# [Function Spaces](FunctionSpaces.md) example. The basis of the form space is obtained by
# multiplying each of them with the basis ``k``-form of its component, numbering first all
# basis forms of the first component, then those of the second one, and so on:
# ```math
# \epsilon^k_i = \hat{B}_{1,i}\,\omega^k_1\,, \quad i = 1, \dots, M_1\,, \qquad
# \epsilon^k_{M_1 + i} = \hat{B}_{2,i}\,\omega^k_2\,, \quad i = 1, \dots, M_2\,, \qquad
# \dots
# ```
# For a ``1``-form in 2D, the function space provides two sets of basis functions,
# ``\{\hat{B}_{1,1}, \dots, \hat{B}_{1,M_1}\}`` and ``\{\hat{B}_{2,1}, \dots,
# \hat{B}_{2,M_2}\}``, and the basis of the form space is
# ```math
# \epsilon^1_1 = \hat{B}_{1,1}\,\mathrm{d}\hat{x}_1\,, \;\dots,\;
# \epsilon^1_{M_1} = \hat{B}_{1,M_1}\,\mathrm{d}\hat{x}_1\,, \;
# \epsilon^1_{M_1+1} = \hat{B}_{2,1}\,\mathrm{d}\hat{x}_2\,, \;\dots,\;
# \epsilon^1_{M_1+M_2} = \hat{B}_{2,M_2}\,\mathrm{d}\hat{x}_2\,.
# ```
# For ``k = 0`` and ``k = n`` there is a single component, so a scalar space is enough:
# ``\epsilon^0_i = \hat{B}_i`` and ``\epsilon^n_i = \hat{B}_i\,\mathrm{d}\hat{x}_1\wedge
# \dots\wedge\mathrm{d}\hat{x}_n``. For ``0 < k < n``, which in 1D, 2D and 3D means
# ``\binom{n}{k} = n`` components, a space with several components is built from scalar
# spaces with `FunctionSpaces.DirectSumSpace((space_1, space_2, ...))`, whose basis
# functions are numbered in exactly this way.
#
# **Form fields.** `Forms.FormField(form_space, coefficients, label)` is the linear
# combination of the basis forms with the given coefficients:
# ```math
# u^k_h = \sum_{i=1}^{M} u_i\,\epsilon^k_i\,, \qquad M = M_1 + M_2 + \dots\,.
# ```
#
# The spaces of the different ranks do not have to be related. Choosing them so that the
# exterior derivative maps each space into the next one gives a de Rham complex, which is
# what makes discretizations stable; see [De Rham Complexes](@ref FormsComplexes). Here we
# only want to show how forms are built, so we use the same B-spline space for every
# component.
#
# On a mapped geometry, the form lives on the manifold ``\mathcal{M}``, and the expressions
# above in ``\hat{\boldsymbol{x}}`` are its representation on the parametric domain, its
# pull-back by the mapping ``\Phi`` of the [Geometry](@ref GeoMapping) example.
#
# **Components in canonical coordinates.** Like geometries and function spaces, forms are
# evaluated element by element at points of the canonical element. `Forms.evaluate` returns
# the components of the form with respect to the canonical basis forms, the table above
# with ``\hat{x}_i`` replaced by ``\xi_i``. On element ``e``, with lower corner
# ``(\hat{x}_{e,1}, \dots, \hat{x}_{e,n})`` and lengths ``h_{e,1}, \dots, h_{e,n}``, the
# element map ``\hat{x}_i = \hat{x}_{e,i} + h_{e,i}\,\xi_i`` gives
# ``\mathrm{d}\hat{x}_i = h_{e,i}\,\mathrm{d}\xi_i``, so the components are multiplied by
# element lengths:
# ```math
# a\,\mathrm{d}\hat{x}_i = a\,h_{e,i}\,\mathrm{d}\xi_i\,, \qquad
# a\,\mathrm{d}\hat{x}_1\wedge\mathrm{d}\hat{x}_2
# = a\,h_{e,1}h_{e,2}\,\mathrm{d}\xi_1\wedge\mathrm{d}\xi_2\,, \qquad \dots
# ```
# A form given in the physical coordinates ``\boldsymbol{x}`` is pulled back with the
# Jacobian ``\mathrm{J}_e`` of ``\Phi_e``. For a ``1``-form and, when ``m = n``, an
# ``n``-form,
# ```math
# \sum_{j=1}^{m} f_j\,\mathrm{d}x_j
# = \sum_{i=1}^{n}\Big(\sum_{j=1}^{m} f_j\,\frac{\partial\Phi_{e,j}}{\partial\xi_i}\Big)
# \mathrm{d}\xi_i\,, \qquad
# f\,\mathrm{d}x_1\wedge\dots\wedge\mathrm{d}x_n
# = f\,\det\mathrm{J}_e\,\mathrm{d}\xi_1\wedge\dots\wedge\mathrm{d}\xi_n\,,
# ```
# that is, the vector of canonical components of a ``1``-form is
# ``\mathrm{J}_e^{\mathsf{T}}\boldsymbol{f}``.
#
# **Proxies.** To look at a form in the physical domain, it is converted to the quantity it
# represents, its **proxy**. With ``\alpha`` the canonical components and ``\mathrm{g}``
# the [metric](@ref GeoMapping) of ``\Phi_e``:
# - a ``0``-form is a function, its own proxy;
# - an ``n``-form ``\alpha\,\mathrm{d}\xi_1\wedge\dots\wedge\mathrm{d}\xi_n`` is
#   represented by its density ``\rho = \alpha/\sqrt{\det\mathrm{g}}``, which is its Hodge
#   star ``\star`` (introduced [below](@ref DFOperations)). On an unmapped geometry,
#   ``\rho = a`` for ``a\,\mathrm{d}x_1\wedge\dots\wedge\mathrm{d}x_n``;
# - a ``1``-form is represented by its **vector proxy**
#   ``\boldsymbol{a} = \mathrm{J}_e\,\mathrm{g}^{-1}\boldsymbol{\alpha}``, given by the
#   sharp operator ``\sharp`` (``\mathrm{g}^{-1}``) followed by a push-forward to the
#   physical domain (``\mathrm{J}_e``). The vector proxy of
#   ``\sum_j a_j\,\mathrm{d}x_j`` is ``(a_1, \dots, a_m)``;
# - in 3D, a ``2``-form is represented by the vector proxy of its Hodge star, a
#   ``1``-form.

# ### [Visualizing forms](@id DFPlotting)
# Besides `Mantis` we use the `LinearAlgebra` and `Printf` standard libraries and, for one
# plot, `GLMakie`.

using Mantis
using LinearAlgebra
using Printf
using GLMakie
using DisplayAs #hide

# `Plot.export_form_fields_to_vtk(forms, filename; output_directory_tree, degree)` writes
# one VTK file per form, named after `filename` and the label of the form, with the proxies
# described above. As in the [Geometry](@ref GeoUnmappedPlot) example, each element is
# sampled with Lagrange cells of degree `degree`. We write the files to a temporary
# folder; replace `output_dir` by any folder of your choice. The figures in this example
# were rendered from these files with ParaView.

output_dir = mktempdir()

# ### [1D: ``0``- and ``1``-forms](@id DFConstruction1D)
# On a line there are only ``0``- and ``1``-forms, and both have a single component: they
# are both built from a scalar function space. We take the [line](@ref GeoLine) ``[0, 2]``
# with 4 elements of the Geometry example,

line = Geometry.create_cartesian_box((0.0,), (2.0,), (4,))
nothing #hide

# and, as in the [Function Spaces](@ref FSUnivariateCode) example, the ``M = 6`` quadratic,
# ``C^1`` B-splines ``\hat{B}_1, \dots, \hat{B}_6`` on it.

B_1D = FunctionSpaces.create_bspline_space(line, (2,), (1,))
nothing #hide

# We use this space for both the ``0``-forms and the ``1``-forms. The geometry is
# unmapped, so ``\hat{x} = x``, and the bases of the two form spaces are
# ```math
# \epsilon^0_i = \hat{B}_i\,, \qquad \epsilon^1_i = \hat{B}_i\,\mathrm{d}x\,, \qquad
# i = 1, \dots, 6\,.
# ```

Λ⁰_1D = Forms.FormSpace(0, B_1D, "Λ⁰")
Λ¹_1D = Forms.FormSpace(1, B_1D, "Λ¹")
nothing #hide

# Both form spaces have one basis form per B-spline:

println("number of basis 0-forms: ", Forms.get_num_basis(Λ⁰_1D))
println("number of basis 1-forms: ", Forms.get_num_basis(Λ¹_1D))

# A form field is a form space together with a vector of coefficients, one per basis form:
# the field is the linear combination
# ```math
# u^0_h = \sum_{i=1}^{6} u_i\,\epsilon^0_i\,, \qquad
# v^1_h = \sum_{i=1}^{6} v_i\,\epsilon^1_i\,.
# ```
# For the ``0``-form we take as coefficients the Greville abscissae of the space, which,
# as shown in the [Function Spaces](@ref FSUnivariateFunctions) example, give
# ``u^0_h = x``. For the ``1``-form we take all coefficients equal to one: by the partition
# of unity of the B-splines, ``\sum_i \hat{B}_i = 1``, so
# ```math
# v^1_h = \Big(\sum_{i=1}^{6} \hat{B}_i\Big)\,\mathrm{d}x = \mathrm{d}x\,.
# ```

(coefficients_u⁰,) = FunctionSpaces.get_greville_points(B_1D)
coefficients_v¹ = ones(Forms.get_num_basis(Λ¹_1D))
u⁰_1D = Forms.FormField(Λ⁰_1D, coefficients_u⁰, "u⁰")
v¹_1D = Forms.FormField(Λ¹_1D, coefficients_v¹, "v¹")
nothing #hide

# `Forms.evaluate(form, element_id, ξ)` evaluates the form field on element `element_id`,
# at a collection of canonical points, ``\xi``. The points are given as one of the point
# types of the [Points](@ref DocPointsModule) module. The simplest one is
# `Points.PointSet`, which takes the coordinates of the points explicitly, one vector per
# direction. Here we take three points of the canonical element ``[0, 1]``:

ξ = Points.PointSet([0.0, 0.5, 1.0])
nothing #hide

# On element 2, ``[0.5, 1]``, the element map is ``x = 0.5 + 0.5\,\xi``. `evaluate`
# returns the canonical components of the form at these points. For the ``0``-form, this
# is its value,
# ```math
# u^0_h\big(x(\xi)\big) = \sum_{i=1}^{6} u_i\,\hat{B}_i\big(x(\xi)\big) = 0.5 + 0.5\,\xi\,.
# ```
# `evaluate` returns two objects. The first one is a vector with one entry per component
# of the form, each a vector with one value per point. The second one gives the indices
# of the evaluated forms. For a form space, these are the indices of the basis forms that
# are non-zero on the element; for a form field there is only one form, the field itself,
# so it is always `[[1]]`.

Forms.evaluate(u⁰_1D, 2, ξ)

# For the ``1``-form, ``v^1_h = \mathrm{d}x = 0.5\,\mathrm{d}\xi`` on element 2, so its
# canonical component is ``0.5`` at every point:

Forms.evaluate(v¹_1D, 2, ξ)

# **Visualization.** For ``0``-forms in 1D, `Mantis` provides a `Makie` plot,
# `Plot.plot_solution` (since `Makie` exports a name `Plot`, we write `Mantis.Plot`):

fig = Mantis.Plot.plot_solution((u⁰_1D,); title=L"u^0_h = x", ylabel=L"u^0_h")
fig = DisplayAs.Text(DisplayAs.PNG(fig)) #hide

# Other forms are exported to VTK files:

Mantis.Plot.export_form_fields_to_vtk(
    (u⁰_1D, v¹_1D), "forms_1D"; output_directory_tree=[output_dir], degree=4
)

# The figure shows the proxies stored in these files: the value of ``u^0_h`` and the
# density of ``v^1_h``, which is 1.
#
# ![The proxies of the 0-form x and the 1-form dx on the line.](../assets/Examples/Forms/forms_1D.png)

# ### [2D: ``0``-, ``1``- and ``2``-forms](@id DFConstruction2D)
# In 2D, ``0``- and ``2``-forms have one component and ``1``-forms have two. We take the
# unit [square](@ref GeoSquare) ``[0, 1]^2`` of the Geometry example, with
# ``4 \times 3`` elements,

square = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (4, 3))
nothing #hide

# and, as in the [Function Spaces](@ref FSTensorProductTheory) example, quadratic, ``C^1``
# B-splines in both directions: 6 univariate B-splines ``\hat{B}^1_i`` in the first
# direction and 5, ``\hat{B}^2_j``, in the second one. Their ``M = 6 \cdot 5 = 30``
# products are numbered with a single index ``l``, with the first direction running
# fastest:
# ```math
# \hat{B}_l = \hat{B}^1_i\,\hat{B}^2_j\,, \qquad l = i + 6(j - 1)\,, \qquad
# i = 1, \dots, 6\,, \quad j = 1, \dots, 5\,.
# ```

B_2D = FunctionSpaces.create_bspline_space(square, (2, 2), (1, 1))
nothing #hide

# We use this space for every component. The ``1``-form space has two components, so its
# function space is the direct sum of two copies of `B_2D`. The bases of the three form
# spaces are, for ``l = 1, \dots, 30``,
# ```math
# \epsilon^0_l = \hat{B}_l\,, \qquad
# \epsilon^1_l = \hat{B}_l\,\mathrm{d}x_1\,,\;\;
# \epsilon^1_{30 + l} = \hat{B}_l\,\mathrm{d}x_2\,, \qquad
# \epsilon^2_l = \hat{B}_l\,\mathrm{d}x_1\wedge\mathrm{d}x_2\,.
# ```

Λ⁰_2D = Forms.FormSpace(0, B_2D, "Λ⁰")
Λ¹_2D = Forms.FormSpace(1, FunctionSpaces.DirectSumSpace((B_2D, B_2D)), "Λ¹")
Λ²_2D = Forms.FormSpace(2, B_2D, "Λ²")
nothing #hide

#-

println("number of basis 0-forms: ", Forms.get_num_basis(Λ⁰_2D))
println("number of basis 1-forms: ", Forms.get_num_basis(Λ¹_2D))
println("number of basis 2-forms: ", Forms.get_num_basis(Λ²_2D))

# We build the form fields
# ```math
# u^0_h = x_1 x_2\,, \qquad
# v^1_h = x_1\,\mathrm{d}x_1 + x_2\,\mathrm{d}x_2\,, \qquad
# w^2_h = (x_1 + x_2)\,\mathrm{d}x_1\wedge\mathrm{d}x_2\,.
# ```
# The Greville abscissae ``\gamma_{1,i}`` and ``\gamma_{2,j}`` of the two directions give
# the coordinates, ``x_1 = \sum_i \gamma_{1,i}\hat{B}^1_i`` and
# ``x_2 = \sum_j \gamma_{2,j}\hat{B}^2_j``. By the partition of unity in each direction,
# the coefficients ``\gamma_{1,i}\,\gamma_{2,j}``, ``\gamma_{1,i}`` and ``\gamma_{2,j}``
# of ``\hat{B}_l`` then give ``x_1 x_2``, ``x_1`` and ``x_2``. So, with
# ``l = i + 6(j - 1)``, ``i = 1, \dots, 6`` and ``j = 1, \dots, 5``,
# ```math
# \begin{aligned}
# u^0_h &= \sum_{l=1}^{30} u_l\,\epsilon^0_l\,, & u_l &= \gamma_{1,i}\,\gamma_{2,j}\,,\\
# v^1_h &= \sum_{l=1}^{60} v_l\,\epsilon^1_l\,, &
# v_l &= \gamma_{1,i}\,, \quad v_{30+l} = \gamma_{2,j}\,,\\
# w^2_h &= \sum_{l=1}^{30} w_l\,\epsilon^2_l\,, & w_l &= \gamma_{1,i} + \gamma_{2,j}\,.
# \end{aligned}
# ```
# In the code, the index ``l`` runs with ``i`` fastest, which is the order in which Julia
# stores a matrix, so the coefficients are vectorized matrices of values at the points
# ``(\gamma_{1,i}, \gamma_{2,j})``.

γ₁, γ₂ = FunctionSpaces.get_greville_points(B_2D)
coefficients_u⁰ = vec([x₁ * x₂ for x₁ in γ₁, x₂ in γ₂])
coefficients_v¹ = vcat(vec([x₁ for x₁ in γ₁, x₂ in γ₂]), vec([x₂ for x₁ in γ₁, x₂ in γ₂]))
coefficients_w² = vec([x₁ + x₂ for x₁ in γ₁, x₂ in γ₂])
u⁰_2D = Forms.FormField(Λ⁰_2D, coefficients_u⁰, "u⁰")
v¹_2D = Forms.FormField(Λ¹_2D, coefficients_v¹, "v¹")
w²_2D = Forms.FormField(Λ²_2D, coefficients_w², "w²")
nothing #hide

# We export them to VTK files:

Mantis.Plot.export_form_fields_to_vtk(
    (u⁰_2D, v¹_2D, w²_2D), "forms_2D"; output_directory_tree=[output_dir], degree=4
)

# The figure shows ``u^0_h``, the vector proxy ``(x_1, x_2)`` of ``v^1_h``, and the density
# of ``w^2_h``.
#
# ![The proxies of the three forms on the square.](../assets/Examples/Forms/forms_2D.png)

# ### [3D: ``0``-, ``1``-, ``2``- and ``3``-forms](@id DFConstruction3D)
# In 3D, ``0``- and ``3``-forms have one component, and ``1``- and ``2``-forms have three.
# We take the unit [cube](@ref GeoCube) ``[0, 1]^3`` of the Geometry example, with
# ``4 \times 3 \times 2`` elements,

cube = Geometry.create_cartesian_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (4, 3, 2))
nothing #hide

# and trilinear, ``C^0`` B-splines on it: 5, 4 and 3 univariate B-splines
# ``\hat{B}^1_i``, ``\hat{B}^2_j`` and ``\hat{B}^3_k`` in the three directions, whose
# Greville abscissae ``\gamma_{1,i}``, ``\gamma_{2,j}`` and ``\gamma_{3,k}`` are the
# breakpoints. Their ``M = 5 \cdot 4 \cdot 3 = 60`` products are numbered with the first
# direction running fastest:
# ```math
# \hat{B}_l = \hat{B}^1_i\,\hat{B}^2_j\,\hat{B}^3_k\,, \qquad
# l = i + 5(j - 1) + 20(k - 1)\,, \qquad
# i = 1, \dots, 5\,, \quad j = 1, \dots, 4\,, \quad k = 1, \dots, 3\,.
# ```

B_3D = FunctionSpaces.create_bspline_space(cube, (1, 1, 1), (0, 0, 0))
nothing #hide

# The ``1``- and ``2``-form spaces use the direct sum of three copies of `B_3D`. With the
# basis forms of the table [above](@ref DFSpacesAndFields), for ``l = 1, \dots, 60``,
# ```math
# \begin{aligned}
# &\epsilon^0_l = \hat{B}_l\,, &
# &\epsilon^1_l = \hat{B}_l\,\mathrm{d}x_1\,,\;
# \epsilon^1_{60+l} = \hat{B}_l\,\mathrm{d}x_2\,,\;
# \epsilon^1_{120+l} = \hat{B}_l\,\mathrm{d}x_3\,,\\
# &\epsilon^3_l = \hat{B}_l\,\mathrm{d}x_1\wedge\mathrm{d}x_2\wedge\mathrm{d}x_3\,, &
# &\epsilon^2_l = \hat{B}_l\,\mathrm{d}x_2\wedge\mathrm{d}x_3\,,\;
# \epsilon^2_{60+l} = \hat{B}_l\,\mathrm{d}x_3\wedge\mathrm{d}x_1\,,\;
# \epsilon^2_{120+l} = \hat{B}_l\,\mathrm{d}x_1\wedge\mathrm{d}x_2\,.
# \end{aligned}
# ```

V_3D = FunctionSpaces.DirectSumSpace((B_3D, B_3D, B_3D))
Λ⁰_3D = Forms.FormSpace(0, B_3D, "Λ⁰")
Λ¹_3D = Forms.FormSpace(1, V_3D, "Λ¹")
Λ²_3D = Forms.FormSpace(2, V_3D, "Λ²")
Λ³_3D = Forms.FormSpace(3, B_3D, "Λ³")
nothing #hide

# As in 2D, we take as coefficients of each component its values at the points
# ``(\gamma_{1,i}, \gamma_{2,j}, \gamma_{3,k})``, which give
# ```math
# \begin{aligned}
# u^0_h &= x_1 x_2 x_3\,, &
# v^1_h &= x_1\,\mathrm{d}x_1 + x_2\,\mathrm{d}x_2 + x_3\,\mathrm{d}x_3\,,\\
# \rho^3_h &= (x_1 + x_2 + x_3)\,\mathrm{d}x_1\wedge\mathrm{d}x_2\wedge\mathrm{d}x_3\,, &
# w^2_h &= x_2\,\mathrm{d}x_2\wedge\mathrm{d}x_3 + x_3\,\mathrm{d}x_3\wedge\mathrm{d}x_1
# + x_1\,\mathrm{d}x_1\wedge\mathrm{d}x_2\,.
# \end{aligned}
# ```

γ₁, γ₂, γ₃ = FunctionSpaces.get_greville_points(B_3D)
x₁ = vec([x₁ for x₁ in γ₁, x₂ in γ₂, x₃ in γ₃])
x₂ = vec([x₂ for x₁ in γ₁, x₂ in γ₂, x₃ in γ₃])
x₃ = vec([x₃ for x₁ in γ₁, x₂ in γ₂, x₃ in γ₃])
u⁰_3D = Forms.FormField(Λ⁰_3D, x₁ .* x₂ .* x₃, "u⁰")
v¹_3D = Forms.FormField(Λ¹_3D, vcat(x₁, x₂, x₃), "v¹")
w²_3D = Forms.FormField(Λ²_3D, vcat(x₂, x₃, x₁), "w²")
ρ³_3D = Forms.FormField(Λ³_3D, x₁ .+ x₂ .+ x₃, "ρ³")
nothing #hide

# We export them to VTK files. The export of ``2``-forms in 3D is not supported yet (it
# writes zeros), so we export the Hodge star of ``w^2_h``, a ``1``-form, whose vector
# proxy is the vector proxy of ``w^2_h``.

Mantis.Plot.export_form_fields_to_vtk(
    (u⁰_3D, v¹_3D, ★(w²_3D), ρ³_3D),
    "forms_3D";
    output_directory_tree=[output_dir],
    degree=4,
)

# The figure shows, from left to right, ``u^0_h``, the vector proxy ``(x_1, x_2, x_3)`` of
# ``v^1_h``, the vector proxy ``(x_2, x_3, x_1)`` of ``w^2_h``, and the density of
# ``\rho^3_h``. The arrows are colored by their length.
#
# ![The proxies of the four forms on the cube.](../assets/Examples/Forms/forms_3D.png)

# ### [Analytical forms](@id DFAnalytical)
# A form can also be given by an analytical expression, for example an exact solution or a
# right-hand side: `Forms.AnalyticalFormField(k, expression, geometry, label)`. The
# argument `expression` is a Julia function that takes the physical points
# ``\boldsymbol{x}^{(1)}, \dots, \boldsymbol{x}^{(P)}``, as a matrix with one row per point
# and one column per coordinate, and returns a vector with one vector of values per
# component of the form, in the physical basis. For example, for the forms
# ```math
# f\,, \qquad \sum_{j=1}^{m} f_j\,\mathrm{d}x_j\,, \qquad
# f\,\mathrm{d}x_1\wedge\dots\wedge\mathrm{d}x_n\,,
# ```
# of rank ``0``, ``1`` and ``n``, it returns ``[f]``, ``[f_1, \dots, f_m]`` and ``[f]``,
# where each ``f_j`` stands for the vector ``(f_j(\boldsymbol{x}^{(1)}), \dots,
# f_j(\boldsymbol{x}^{(P)}))``. Here ``m`` is the image dimension of the geometry, and
# ``n``-forms require ``m = n``. Analytical ``2``-forms in 3D are not supported yet.
#
# The expression is defined in physical coordinates, but, like all forms in `Mantis`, an
# analytical form is evaluated in the canonical domain: when it is evaluated on an
# element, `Mantis` evaluates the expression at the physical points ``\Phi_e(\xi)`` and
# pulls the result back automatically, with the formulas of
# [Components in canonical coordinates](@ref DFSpacesAndFields).
#
# As an example, the ``1``-form ``v^1 = x_1\,\mathrm{d}x_1 + x_2\,\mathrm{d}x_2`` on the
# square has ``(f_1, f_2) = (x_1, x_2)``:

v¹_expression(x) = [x[:, 1], x[:, 2]]
v¹_exact = Forms.AnalyticalFormField(1, v¹_expression, square, "v¹ exact")
nothing #hide

# Since the B-splines reproduce linear functions, the finite element form ``v^1_h`` of the
# previous section is equal to it, at any point. To check this on a grid of points, we use
# `Points.TensorProductPoints`, which takes one vector of coordinates per direction and
# forms all their combinations; see [TensorProductPoints](@ref PointsTensorProduct). Here
# it gives ``3 \times 2`` points on each element:

ξ_2D = Points.TensorProductPoints(([0.0, 0.3, 1.0], [0.2, 0.9]))
all(
    Forms.evaluate(v¹_exact, e, ξ_2D)[1] ≈ Forms.evaluate(v¹_2D, e, ξ_2D)[1] for
    e in 1:Geometry.get_num_elements(square)
)

# Analytical forms are evaluated and exported like any other form.

# ## [``L^2`` projection](@id DFProjection)
# ### [The projection](@id DFProjectionTheory)
# The ``L^2`` inner product of two ``k``-forms on a domain ``\Omega`` is
# ```math
# (\alpha^k, \beta^k)_\Omega = \int_\Omega \alpha^k\wedge\star\beta^k\,.
# ```
# The ``L^2`` projection of a ``k``-form ``f^k`` onto a finite element form space
# ``\Lambda^k_h`` is the form ``f^k_h \in \Lambda^k_h`` closest to ``f^k`` in the norm of
# this inner product. It is characterized by
# ```math
# (v^k, f^k_h)_\Omega = (v^k, f^k)_\Omega \qquad \forall v^k \in \Lambda^k_h\,.
# ```
# Writing ``f^k_h = \sum_{j=1}^{M} c_j\,\epsilon^k_j`` and taking ``v^k = \epsilon^k_i``
# gives a linear system for the coefficients,
# ```math
# \sum_{j=1}^{M} \mathsf{M}_{ij}\,c_j = b_i\,, \qquad
# \mathsf{M}_{ij} = (\epsilon^k_i, \epsilon^k_j)_\Omega\,, \qquad
# b_i = (\epsilon^k_i, f^k)_\Omega\,, \qquad i = 1, \dots, M\,,
# ```
# whose matrix ``\mathsf{M}`` is the mass matrix of ``\Lambda^k_h``.
# `Assemblers.solve_L2_projection(Λ, f, dΩ)` assembles and solves this system with the
# quadrature rule `dΩ`, and returns ``f^k_h`` as a `FormField`.

# ### [Two forms on the unit square](@id DFProjectionSquare)
# We project the ``0``-form and the ``1``-form
# ```math
# f^0 = \sin(\pi x_1)\sin(\pi x_2)\,, \qquad
# f^1 = \sin(\pi x_1)\cos(\pi x_2)\,\mathrm{d}x_1
# - \cos(\pi x_1)\sin(\pi x_2)\,\mathrm{d}x_2\,,
# ```
# first on a coarse mesh, with linear B-splines on the unit square with ``2 \times 2``
# elements.

coarse_square = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (2, 2))
nothing #hide

# The two forms are analytical forms on this geometry:

f⁰_expression(x) = [@. sinpi(x[:, 1]) * sinpi(x[:, 2])]
f¹_expression(x) =
    [@.(sinpi(x[:, 1]) * cospi(x[:, 2])), @.(-cospi(x[:, 1]) * sinpi(x[:, 2]))]
f⁰_coarse = Forms.AnalyticalFormField(0, f⁰_expression, coarse_square, "f⁰")
f¹_coarse = Forms.AnalyticalFormField(1, f¹_expression, coarse_square, "f¹")
nothing #hide

# We create a B-spline space that is linear (``p = 1``) and ``C^0`` in both directions,

B_coarse = FunctionSpaces.create_bspline_space(coarse_square, (1, 1), (0, 0))
nothing #hide

# and the form spaces are built from it as in the [2D section](@ref DFConstruction2D):

Λ⁰_coarse = Forms.FormSpace(0, B_coarse, "Λ⁰")
Λ¹_coarse = Forms.FormSpace(1, FunctionSpaces.DirectSumSpace((B_coarse, B_coarse)), "Λ¹")
nothing #hide

# The integrals are computed with a Gauss–Legendre quadrature rule with 4 points per
# direction on every element; see the [Quadrature](@ref) documentation:

rule = Quadrature.tensor_product_rule((4, 4), Quadrature.gauss_legendre)
dΩ_coarse = Quadrature.StandardQuadrature(rule, Geometry.get_num_elements(coarse_square))
nothing #hide

# The projections are

f⁰_h_coarse = Assemblers.solve_L2_projection(Λ⁰_coarse, f⁰_coarse, dΩ_coarse)
f¹_h_coarse = Assemblers.solve_L2_projection(Λ¹_coarse, f¹_coarse, dΩ_coarse)
nothing #hide

# `Analysis.compute_error_total(f_h, f, dΩ)` computes the ``L^2`` norm of the difference,
# ```math
# \|f^k_h - f^k\|_{L^2} = (f^k_h - f^k, f^k_h - f^k)_\Omega^{1/2}\,,
# ```
# with the quadrature rule `dΩ`. Both forms must be defined on the same geometry, which is
# why we built the analytical forms on `coarse_square`.

error_f⁰_coarse = Analysis.compute_error_total(f⁰_h_coarse, f⁰_coarse, dΩ_coarse)
error_f¹_coarse = Analysis.compute_error_total(f¹_h_coarse, f¹_coarse, dΩ_coarse)
@printf("coarse: L2 error of f⁰: %.2e, of f¹: %.2e\n", error_f⁰_coarse, error_f¹_coarse)

# We repeat the same steps on a fine mesh, with cubic (``p = 3``), ``C^2`` B-splines on
# ``12 \times 12`` elements and 6 quadrature points per direction.

fine_square = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (12, 12))
f⁰_fine = Forms.AnalyticalFormField(0, f⁰_expression, fine_square, "f⁰")
f¹_fine = Forms.AnalyticalFormField(1, f¹_expression, fine_square, "f¹")
B_fine = FunctionSpaces.create_bspline_space(fine_square, (3, 3), (2, 2))
Λ⁰_fine = Forms.FormSpace(0, B_fine, "Λ⁰")
Λ¹_fine = Forms.FormSpace(1, FunctionSpaces.DirectSumSpace((B_fine, B_fine)), "Λ¹")
rule = Quadrature.tensor_product_rule((6, 6), Quadrature.gauss_legendre)
dΩ_fine = Quadrature.StandardQuadrature(rule, Geometry.get_num_elements(fine_square))
f⁰_h_fine = Assemblers.solve_L2_projection(Λ⁰_fine, f⁰_fine, dΩ_fine)
f¹_h_fine = Assemblers.solve_L2_projection(Λ¹_fine, f¹_fine, dΩ_fine)
error_f⁰_fine = Analysis.compute_error_total(f⁰_h_fine, f⁰_fine, dΩ_fine)
error_f¹_fine = Analysis.compute_error_total(f¹_h_fine, f¹_fine, dΩ_fine)
@printf("fine:   L2 error of f⁰: %.2e, of f¹: %.2e\n", error_f⁰_fine, error_f¹_fine)

# We export the exact forms and the two projections. The label of a form is the name of its
# field in the VTK file, as shown in ParaView, and is also appended to the file name.
# The coarse and fine projections have the same labels, inherited from their form spaces,
# so we export them with different file names.

Mantis.Plot.export_form_fields_to_vtk(
    (f⁰_fine, f¹_fine), "projection_exact"; output_directory_tree=[output_dir], degree=4
)
Mantis.Plot.export_form_fields_to_vtk(
    (f⁰_h_coarse, f¹_h_coarse),
    "projection_coarse";
    output_directory_tree=[output_dir],
    degree=4,
)
Mantis.Plot.export_form_fields_to_vtk(
    (f⁰_h_fine, f¹_h_fine), "projection_fine"; output_directory_tree=[output_dir], degree=4
)

# The figures compare the exact forms with the two projections. The coarse projection of
# ``f^0`` is bilinear on each element and misses the round peak, and the coarse ``1``-form
# only follows the vortex roughly. The fine projections cannot be told apart from the exact
# forms.
#
# ![The 0-form: exact, coarse and fine projections.](../assets/Examples/Forms/projection_f0.png)
#
# ![The vector proxy of the 1-form: exact, coarse and fine projections.](../assets/Examples/Forms/projection_f1.png)

# ### [A curved geometry](@id DFProjectionAnnulus)
# Nothing changes on a mapped geometry: the analytical forms are given in physical
# coordinates and `Mantis` pulls them back. On the [half annulus](@ref GeoAnnulus) of the
# Geometry example, with quadratic, ``C^1`` B-splines on ``4 \times 12`` elements, we
# repeat the same steps with the same expressions, now evaluated on
# ``1 \leq |\boldsymbol{x}| \leq 2``, ``x_2 \geq 0``.

annulus = Geometry.create_annulus_sector((4, 12))
f⁰_annulus = Forms.AnalyticalFormField(0, f⁰_expression, annulus, "f⁰")
f¹_annulus = Forms.AnalyticalFormField(1, f¹_expression, annulus, "f¹")
B_annulus = FunctionSpaces.create_bspline_space(annulus, (2, 2), (1, 1))
Λ⁰_annulus = Forms.FormSpace(0, B_annulus, "Λ⁰")
Λ¹_annulus = Forms.FormSpace(1, FunctionSpaces.DirectSumSpace((B_annulus, B_annulus)), "Λ¹")
rule = Quadrature.tensor_product_rule((5, 5), Quadrature.gauss_legendre)
dΩ_annulus = Quadrature.StandardQuadrature(rule, Geometry.get_num_elements(annulus))
f⁰_h_annulus = Assemblers.solve_L2_projection(Λ⁰_annulus, f⁰_annulus, dΩ_annulus)
f¹_h_annulus = Assemblers.solve_L2_projection(Λ¹_annulus, f¹_annulus, dΩ_annulus)
error_f⁰_annulus = Analysis.compute_error_total(f⁰_h_annulus, f⁰_annulus, dΩ_annulus)
error_f¹_annulus = Analysis.compute_error_total(f¹_h_annulus, f¹_annulus, dΩ_annulus)
@printf("annulus: L2 error of f⁰: %.2e, of f¹: %.2e\n", error_f⁰_annulus, error_f¹_annulus)

#-

Mantis.Plot.export_form_fields_to_vtk(
    (f⁰_annulus, f¹_annulus), "annulus_exact"; output_directory_tree=[output_dir], degree=4
)
Mantis.Plot.export_form_fields_to_vtk(
    (f⁰_h_annulus, f¹_h_annulus),
    "annulus_projection";
    output_directory_tree=[output_dir],
    degree=4,
)

# The figure shows the exact ``0``-form and its projection (top), and the vector proxies
# of the exact ``1``-form and of its projection (bottom).
#
# ![The exact forms and their projections on the half annulus.](../assets/Examples/Forms/projection_annulus.png)

# ## [Operations on forms](@id DFOperations)
# ### [Exterior derivative, Hodge star and wedge product](@id DFOperationsTheory)
# The `Forms` module provides the operators of exterior calculus; see
# [Operations on Forms](@ref FormsOperations). `Mantis` exports the most common ones, so
# they can be used without the `Forms.` prefix. We use three of them:
# - the **exterior derivative** `d`, which maps a ``k``-form to a ``(k+1)``-form. It does
#   not depend on the metric, so it has the same expression in any coordinates. `Mantis`
#   applies it to the canonical components; in 2D,
#   ```math
#   \mathrm{d}\alpha = \frac{\partial\alpha}{\partial\xi_1}\,\mathrm{d}\xi_1
#   + \frac{\partial\alpha}{\partial\xi_2}\,\mathrm{d}\xi_2\,, \qquad
#   \mathrm{d}(\alpha_1\,\mathrm{d}\xi_1 + \alpha_2\,\mathrm{d}\xi_2)
#   = \Big(\frac{\partial\alpha_2}{\partial\xi_1} - \frac{\partial\alpha_1}{\partial\xi_2}
#   \Big)\,\mathrm{d}\xi_1\wedge\mathrm{d}\xi_2\,.
#   ```
#   In terms of proxies, it is the gradient on ``0``-forms and the scalar curl on
#   ``1``-forms. For a finite element form it acts on the basis forms, for example
#   ``\mathrm{d}u^0_h = \sum_i u_i\,\mathrm{d}\epsilon^0_i`` with
#   ``\mathrm{d}\epsilon^0_i = \sum_j \frac{\partial\hat{B}_i}{\partial\hat{x}_j}\,
#   \mathrm{d}\hat{x}_j``;
# - the **Hodge star** `★` (typed `\bigstar`), which maps a ``k``-form to an
#   ``(n-k)``-form and depends on the metric ``\mathrm{g}``, with entries ``g_{ij}`` and
#   inverse ``\mathrm{g}^{-1}`` with entries ``g^{ij}``. In 2D, in canonical components,
#   ```math
#   \begin{aligned}
#   \star\alpha &= \alpha\sqrt{\det\mathrm{g}}\;\mathrm{d}\xi_1\wedge\mathrm{d}\xi_2\,,
#   \qquad
#   \star(\alpha\,\mathrm{d}\xi_1\wedge\mathrm{d}\xi_2)
#   = \frac{\alpha}{\sqrt{\det\mathrm{g}}}\,,\\
#   \star(\alpha_1\,\mathrm{d}\xi_1 + \alpha_2\,\mathrm{d}\xi_2)
#   &= \sqrt{\det\mathrm{g}}\,\Big(-\big(g^{21}\alpha_1 + g^{22}\alpha_2\big)\,
#   \mathrm{d}\xi_1 + \big(g^{11}\alpha_1 + g^{12}\alpha_2\big)\,\mathrm{d}\xi_2\Big)\,.
#   \end{aligned}
#   ```
#   On the unit square, in the physical coordinates, these reduce to
#   ``\star f = f\,\mathrm{d}x_1\wedge\mathrm{d}x_2`` and
#   ``\star(a_1\,\mathrm{d}x_1 + a_2\,\mathrm{d}x_2) = -a_2\,\mathrm{d}x_1 +
#   a_1\,\mathrm{d}x_2``, a rotation of the vector proxy by ``90^\circ``;
# - the **wedge product** `∧` (typed `\wedge`), which maps a ``k``-form and an
#   ``l``-form to a ``(k+l)``-form. It does not depend on the metric. In 2D,
#   ```math
#   f\wedge(a_1\,\mathrm{d}x_1 + a_2\,\mathrm{d}x_2)
#   = f a_1\,\mathrm{d}x_1 + f a_2\,\mathrm{d}x_2\,, \qquad
#   (a_1\,\mathrm{d}x_1 + a_2\,\mathrm{d}x_2)\wedge(b_1\,\mathrm{d}x_1 + b_2\,\mathrm{d}x_2)
#   = (a_1 b_2 - a_2 b_1)\,\mathrm{d}x_1\wedge\mathrm{d}x_2\,,
#   ```
#   and the same expressions hold for the canonical components.
#
# These operators do not compute anything when they are applied: `d(f⁰_h)` is an
# expression, which is evaluated when it is evaluated, exported or integrated.
# Expressions can be combined with each other and with `+`, `-` and multiplication by
# numbers. The exterior derivative of an analytical form is not available, so we give the
# exact results as analytical forms.
#
# For the forms of the previous section, the exact results are
# ```math
# \begin{aligned}
# \mathrm{d}f^0 &= \pi\cos(\pi x_1)\sin(\pi x_2)\,\mathrm{d}x_1
# + \pi\sin(\pi x_1)\cos(\pi x_2)\,\mathrm{d}x_2\,, &
# \mathrm{d}f^1 &= 2\pi\sin(\pi x_1)\sin(\pi x_2)\,\mathrm{d}x_1\wedge\mathrm{d}x_2\,,\\
# \star f^0 &= \sin(\pi x_1)\sin(\pi x_2)\,\mathrm{d}x_1\wedge\mathrm{d}x_2\,, &
# \star f^1 &= \cos(\pi x_1)\sin(\pi x_2)\,\mathrm{d}x_1
# + \sin(\pi x_1)\cos(\pi x_2)\,\mathrm{d}x_2\,,\\
# f^0\wedge f^1 &= \sin(\pi x_1)\sin(\pi x_2)\,\big(\sin(\pi x_1)\cos(\pi x_2)\,
# \mathrm{d}x_1 - \cos(\pi x_1)\sin(\pi x_2)\,\mathrm{d}x_2\big)\,, &
# f^1\wedge\mathrm{d}f^0 &= \pi\big(\sin^2(\pi x_1)\cos^2(\pi x_2)
# + \cos^2(\pi x_1)\sin^2(\pi x_2)\big)\,\mathrm{d}x_1\wedge\mathrm{d}x_2\,.
# \end{aligned}
# ```

df⁰_expression(x) =
    [@.(π * cospi(x[:, 1]) * sinpi(x[:, 2])), @.(π * sinpi(x[:, 1]) * cospi(x[:, 2]))]
df¹_expression(x) = [@. 2π * sinpi(x[:, 1]) * sinpi(x[:, 2])]
star_f¹_expression(x) =
    [@.(cospi(x[:, 1]) * sinpi(x[:, 2])), @.(sinpi(x[:, 1]) * cospi(x[:, 2]))]
f⁰f¹_expression(x) = [f⁰_expression(x)[1] .* component for component in f¹_expression(x)]
f¹_wedge_df⁰_expression(x) =
    [@.(π * (sinpi(x[:, 1])^2 * cospi(x[:, 2])^2 + cospi(x[:, 1])^2 * sinpi(x[:, 2])^2))]
nothing #hide

# ### [Comparing with the exact results](@id DFOperationsChecks)
# We apply each operation to the coarse and fine projections ``f^0_h`` and ``f^1_h`` of
# the previous section, and compute the ``L^2`` error with respect to the exact result.
# Each operation is given by its name, the expression applied to the projections, and the
# rank and expression of the exact result.

for (mesh, geometry, f⁰_h, f¹_h, dΩ) in (
    ("coarse", coarse_square, f⁰_h_coarse, f¹_h_coarse, dΩ_coarse),
    ("fine", fine_square, f⁰_h_fine, f¹_h_fine, dΩ_fine),
)
    println(mesh)
    for (name, computed, rank, expression) in (
        ("d f⁰", d(f⁰_h), 1, df⁰_expression),
        ("d f¹", d(f¹_h), 2, df¹_expression),
        ("★ f⁰", ★(f⁰_h), 2, f⁰_expression),
        ("★ f¹", ★(f¹_h), 1, star_f¹_expression),
        ("f⁰ ∧ f¹", f⁰_h ∧ f¹_h, 1, f⁰f¹_expression),
        ("f¹ ∧ d f⁰", f¹_h ∧ d(f⁰_h), 2, f¹_wedge_df⁰_expression),
    )
        exact = Forms.AnalyticalFormField(rank, expression, geometry, name)
        error = Analysis.compute_error_total(computed, exact, dΩ)
        @printf("  %-10s L2 error: %.2e\n", name, error)
    end
end

# All errors decrease under refinement. The results that involve an exterior derivative
# (`d f⁰`, `d f¹` and `f¹ ∧ d f⁰`) have larger errors, because they contain derivatives of
# the projections.
#
# Expressions are exported like any other form. We export three of the computed forms on
# the fine mesh: the gradient ``\mathrm{d}f^0_h``, the curl ``\mathrm{d}f^1_h`` and
# ``f^1_h\wedge\mathrm{d}f^0_h``.

Mantis.Plot.export_form_fields_to_vtk(
    (d(f⁰_h_fine), d(f¹_h_fine), f¹_h_fine ∧ d(f⁰_h_fine)),
    "operations";
    output_directory_tree=[output_dir],
    degree=4,
)

# The figure shows the vector proxy of ``\mathrm{d}f^0_h``, and the densities of
# ``\mathrm{d}f^1_h`` and ``f^1_h\wedge\mathrm{d}f^0_h``.
#
# ![The results of three operations on the fine projections.](../assets/Examples/Forms/operations.png)

# ## [Summary and outlook](@id DFSummary)
# In this example we learned that:
# - a form space `Forms.FormSpace(k, space, label)` combines a function space with
#   ``\binom{n}{k}`` components with the basis ``k``-forms, and a form field
#   `Forms.FormField(form_space, coefficients, label)` is a linear combination of its basis
#   forms. Spaces with several components are direct sums of scalar spaces;
# - `Forms.AnalyticalFormField(k, expression, geometry, label)` defines a form by an
#   expression in physical coordinates;
# - forms are evaluated in canonical coordinates, and visualized through their proxies:
#   values, densities and vector proxies;
# - `Assemblers.solve_L2_projection` computes the ``L^2`` projection of a form onto a form
#   space, and `Analysis.compute_error_total` its ``L^2`` error;
# - `d`, `★` and `∧` build expressions that are evaluated, exported and integrated like
#   forms.
#
# The functions used in this example are:
#
# | task | function |
# |:--- |:--- |
# | form space | `Forms.FormSpace(k, space, label)` |
# | multi-component space | `FunctionSpaces.DirectSumSpace(spaces)` |
# | form field | `Forms.FormField(form_space, coefficients, label)` |
# | analytical form | `Forms.AnalyticalFormField(k, expression, geometry, label)` |
# | evaluation | `Forms.evaluate` |
# | VTK export | `Plot.export_form_fields_to_vtk` |
# | ``0``-form plot in 1D | `Plot.plot_solution` |
# | ``L^2`` projection | `Assemblers.solve_L2_projection` |
# | ``L^2`` error | `Analysis.compute_error_total` |
# | operations | `d`, `★`, `∧` |
