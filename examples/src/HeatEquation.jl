# # Heat Equation

# ## Background knowledge

# ### A. The Heat Equation in 1D

# #### 1. The differential equation

# The heat equation in one dimension is given by:
# ```math
# \frac{\partial u}{\partial t} - \frac{\partial}{\partial x} \left( \alpha
# \frac{\partial u}{\partial x} \right) = f \quad \text{for} \quad x \in (0, L)
# \quad \text{and} \quad t > 0
# ```
# where:
# - ``x`` is the spatial coordinate,
# - ``t`` is the time,
# - ``\alpha(x)`` is the spatially-varying thermal diffusivity of the material,
# - ``u(x, t)`` is the temperature distribution function,
# - ``f(x, t)`` is a time dependent and spatially varying heat source.

# #### 2. Initial and boundary conditions

# To solve the heat equation, we must specify initial and boundary conditions:
# 1. **Initial Condition**: This refers to the temperature distribution at the initial time
#    ``t = 0``. We need to know this temperature distribution in our entire domain
#    ``(0, L)``.
#    ```math
#    u(x, 0) = u_0(x)
#    ```
# 2. **Boundary Conditions**: We can specify different types of boundary conditions, i.e.,
#    conditions on the boundaries of our domain, ``x = 0`` and ``x = L``.
#    - **Dirichlet Boundary Conditions**: This boundary condition means that the temperature
#      is specified at the boundaries of the domain.
#      ```math
#      u(0, t) = u_0 \quad \text{and} \quad u(L, t) = u_L
#      ```
#    - **Neumann Boundary Conditions**: This boundary condition means that, instead of the
#      temperature, the derivative of the temperature (which is called the heat flux) is
#      specified at the boundaries of the domain.
#      ```math
#      \alpha(0)\frac{\partial u}{\partial x}(0, t) = g_0 \quad \text{and} \quad \alpha(L)
#      \frac{\partial u}{\partial x}(L, t) = g_L
#      ```

# #### 3. Applications

# A simplified example where the heat equation can be used is to find out how the
# temperature is distributed through the outside, insulating walls of your apartment. Look
# at the image below
# ([source](https://www.anglian-building.co.uk/products/external-wall-insulation/)).
# ![wall-insulation](../assets/wall-insulation.jpg) #md
# ![wall-insulation](../../docs/src/assets/wall-insulation.jpg) #nb
#
# The insulation wall is made up of several materials, each with their own thermal
# diffusivities ``\alpha_i(x)``. Imagine that the temperature outside is 0 degrees, and your
# heating system holds the temperature inside your house at 18 degrees. Then, these are the
# boundary conditions for the heat equation. Given an initial temperature distribution
# through the insulation wall, you could use the heat equation to find out how the
# temperature varies inside the insulation wall.

# ### B. Solving the Heat Equation

# #### 1. Disadvantages of the above formulation

# Now, if we want to compute the solution of the heat equation as stated above, we run into
# two difficulties:
# 1. **Lack of exact solutions**: For a general function ``f``, finding out the exact,
#    analytical solution ``u`` is not an easy task. Well, this is not entirely true:
#    finding the solution can be easy enough in one dimension on a domain as simple as
#    ``(0, L)``, but in higher dimensions and on more complicated geometries (e.g., imagine
#    the insulation wall of the
#    [Guggenheim museum](https://www.britannica.com/topic/Guggenheim-Museum-Bilbao)) it is
#    not possible.
# 2. **A restrictive set of solutions**: For the above form of the heat equation to make
#    sense, we must also assume that the second-derivatives of the solution ``u(x)`` should
#    exist, and that the first derivatives of the thermal diffusivity ``\alpha(x)`` should
#    exist. It turns out that this is too *strong* of a requirement that it not satisfied
#    by many physical systems.
#
# For example, think of the insulation wall - each material in the insulation wall has its
# own thermal diffusivity which is completely unrelated to the diffusivities of the other
# materials. As a result, ``\alpha(x)`` is a discontinuous function and its first
# derivatives do not make sense.

# #### 2. Tackling the above disadvantages using a discrete & weaker formulation

# The above disadvantages are the reason why, in practice, the above *strong* formulation
# of the heat equation is not useful. Instead, we formulate a *discrete, weak* version of
# the equation which is much more useful in practice. The motivation is:
# 1. **Discrete approximation of unknown exact solutions**: Since we don't know the exact
#    solution in general, we try to approximate it. This is the process called
#    discretization. In this process, we fix a finite-dimensional vector space of
#    *spatially-varying* functions ``V_n`` and say that, for any given time ``t``, we want
#    to find a function ``u_n(\cdot,t) \in V_n`` which approximates the exact solution
#    ``u(\cdot,t)``. Here, ``n`` denotes the dimension of the vector space ``V_n``. We
#    expect that as ``n \rightarrow \infty``, the solution
#    ``u_n(\cdot,t) \rightarrow u(\cdot,t)``.
# 2. **Weak version of the equation**: Since the original equation imposes too strong
#    requirements on the smoothness of ``u(x,t)`` and ``\alpha(x)``, we instead work with
#    an integral formulation where only the first derivatives of ``u(x,t)`` should make
#    sense, and where ``\alpha(x)`` is allowed to be discontinuous.
#
# > **_ASSUMPTION:_**  For the sake of simplifying the discussion, from now on we will
# > assume that we are imposing Dirichlet boundary conditions at ``x = 0`` and ``x = L``.
#
# This discrete, weak version of the problem at a *fixed time* ``t`` is stated as: find
# ``u_n(\cdot,t) \in S_n`` such that
# ```math
# \int_{0}^{L} w_n\frac{\partial u_n}{\partial t}\;\mathrm{d}x + \int_{0}^{L} \alpha
# \frac{\partial u_n}{\partial x}\frac{\partial w_n}{\partial x} \,\mathrm{d}x = \int_0^L
# f w_n \, dx\,,\qquad \forall w_n\in W_n\;,
# ```
# where:
# * ``S_n := \{v_n(x) \in V_n~:~v_n(0) = u_0\;,\;v_n(L) = u_L\}``,
# * ``W_n := \{v_n(x) \in V_n~:~v_n(0) = 0\;,\;v_n(L) = 0\}``.
#
# Note the following important things:
# 1. The above problem tries to find the solution ``u_n(\cdot,t)`` at the *fixed* time
#    instant ``t``.
# 2. ``S_n`` consists of spatially-varying functions in ``V_n`` that satisfy the boundary
#    conditions.
# 3. We want the integral equation above to be satisfied for all functions ``w_n \in W_n``,
#    where the function space ``W_n`` consists of spatially-varying functions in ``V_n``
#    that satisfy homogeneous (or, equivalently, zero) boundary conditions.


# ## The finite element method

# Now we are at a stage where, if we make a choice for ``V_n``, we can convert the discrete
# weak problem into a system of ODEs. This section will explain how.

# ### A. How to choose ``V_n``?

# #### 1. Choosing ``V_n``

# In *the finite element method*, we choose ``V_n`` as the space of piecewise-polynomial
# functions of degree ``p`` on a mesh of the domain ``(0, L)``.

# ##### Meshing the domain
# We choose a set of ``N+1`` points, ``0 = x_1 < x_1 < x_2 < \cdots < x_{N+1} = L``, and
# these points divide the domain ``(0, L)`` into smaller subdomains, ``(x_{i}, x_{i+1})``
# with ``x_{i}\in (0, L)``, called *elements*. That is, we assume that there are ``N``
# elements in our mesh.
#
# > **_ASSUMPTION:_** In the following code, we will always assume that the mesh is
# > *uniform*. In other words, ``x_{i+1}-x_i`` is equal to ``L/N`` for all ``i``.

# ##### Defining ``V_n``

# The space ``V_n`` is defined as the space of functions ``v_n`` such that:
# * on any element (i.e., on the interval ``(x_{i}, x_{i+1})`` ``with i = 1, \dots, N``) it
#   is a polynomial of degree ``p``,
# * at each ``x_i``, ``i = 2, \dots, N``, the function ``v_n`` is ``C^k`` smooth for some
#   ``k \geq 0``.
#
# Once we do this, the vector-space dimension of ``V_n`` can be related to the parameters
# ``N, p, k`` as follows:
# ```math
#  n = (p+1)N - (k+1)(N-1)\;.
# ```
# In other words, there are ``n`` basis functions ``\phi_{i}(x)``, ``i = 1, \dots, n``,
# such that any arbitrary ``v_n \in V_n`` can be represented as:
# ```math
# v_n(x) = \sum_{i=1}^n c_i \phi_{i}(x)\;,
# ```
# for some numbers ``c_i \in \mathbb{R}``.
#
# > **EXAMPLE:** ``V_n`` with ``(N,p,k) = (4, 1, 0)``.
# Consider the space of functions that are linear polynomials over each mesh element, and
# which are ``C^0`` smooth (or, equivalently, continuous) at the interfaces ``x_i``
# between the elements. This space of functions has dimension:
# ```math
# n = (p+1)N - (k+1)(N-1) = 2\times 4 - 1 \times 3 = 5\;.
# ```
# So, we can find 5 basis functions, ``\phi_{1}, \phi_{2}, \dots \phi_{5}``, that span the
# space ``V_n``. Run the code below to create such a ``V_n`` and look at one such choice of
# the basis functions called *hat functions*. Convince yourself that linear combinations of
# these functions can be used to represent any piecewise-linear polynomial function on the
# mesh. (Each function is plotted in a different color.)

import Mantis
import DisplayAs
using CairoMakie

## The size of the domain where to solve our problem
L = 1.0

## The degree of the piecewise-polynomial basis functions
p = 1
## The number of elements in the mesh
N = 4
## The smoothness of the basis functions (must be smaller than the polynomial degree, and
## larger than -1)
k = 0

## The number of basis functions in the piecewise-polynomial function space
n = N * (p + 1) - (k + 1) * (N - 1)

## Create the mesh and the function space
breakpoints = LinRange(0.0, L, N+1)
patch = Mantis.Mesh.Patch1D(collect(breakpoints))
B = Mantis.FunctionSpaces.BSplineSpace(patch, p, k)
line_geo = Mantis.Geometry.CartesianGeometry((breakpoints,))

## Create a Form Space.
BF = Mantis.Forms.FormSpace(0, line_geo, B, "b")

## Plot the basis functions.
n_plot_points_per_element = 25

fig = Figure()
ax = Axis(fig[1, 1],
    title = "Basis functions of V_n",
    xlabel = "x",
    ylabel = "b_i(x)",
)

n_elements = Mantis.Geometry.get_num_elements(line_geo)
xi = Mantis.Points.CartesianPoints((LinRange(0.0, 1.0, n_plot_points_per_element),))
BFF = Mantis.Forms.FormField(BF, " ")

dim_V = Mantis.Forms.get_num_basis(BF)
colors = [:blue, :green, :red, :purple, :orange]
for basis_idx in 1:dim_V

    BFF.coefficients[basis_idx] = 1.0
    if basis_idx > 1
        BFF.coefficients[basis_idx - 1] = 0.0
    end

    color_i = colors[basis_idx]

    for element_idx in 1:n_elements
        form_eval, _ = Mantis.Forms.evaluate(BFF, element_idx, xi)
        x = Mantis.Geometry.evaluate(Mantis.Forms.get_geometry(BF), element_idx, xi)

        lines!(ax, x[:], form_eval[1], color=color_i, label=L"\phi_{%$basis_idx}")

        scatter!(ax, x[:][[1, end]], [0.0, 0.0], color=:tomato)
    end
end
fig[1, 2] = Legend(fig, ax, marge=true, unique=true)

fig = DisplayAs.Text(DisplayAs.PNG(fig))

# > **EXAMPLE:** ``V_n`` with ``(N,p,k) = (4, 2, 1)``.
# Consider now the space of functions that are quadratic polynomials over each mesh
# element, and which are ``C^1`` smooth (or, equivalently, continuous and continuously
# differentiable) at the interfaces ``x_i`` between the elements. This space of functions
# has dimension:
# ```math
# n = (p+1)N - (k+1)(N-1) = 3\times 4 - 2 \times 3 = 6\;.
# ```
# So, we can find 6 basis functions, ``\phi_{1}, \phi_{2}, \dots \phi_{6}``, that span the
# space ``V_n``. Run the code below to create such a ``V_n`` and look at one such choice of
# the basis functions called *B-splines*. (Each function is plotted in a different color.)


p = 2
k = 1
n = N * (p + 1) - (k + 1) * (N - 1)

B = Mantis.FunctionSpaces.BSplineSpace(patch, p, k)

## Create a Form Space.
BF = Mantis.Forms.FormSpace(0, line_geo, B, "b")

fig = Figure()
ax = Axis(fig[1, 1],
    title = "Basis functions of V_n",
    xlabel = "x",
    ylabel = "b_i(x)",
)

n_elements = Mantis.Geometry.get_num_elements(line_geo)
xi = Mantis.Points.CartesianPoints((LinRange(0.0, 1.0, n_plot_points_per_element),))
BFF = Mantis.Forms.FormField(BF, " ")

dim_V = Mantis.Forms.get_num_basis(BF)
colors = [:blue, :green, :red, :purple, :orange, :black]
for basis_idx in 1:dim_V

    BFF.coefficients[basis_idx] = 1.0
    if basis_idx > 1
        BFF.coefficients[basis_idx - 1] = 0.0
    end

    color_i = colors[basis_idx]

    for element_idx in 1:n_elements
        form_eval, _ = Mantis.Forms.evaluate(BFF, element_idx, xi)
        x = Mantis.Geometry.evaluate(Mantis.Forms.get_geometry(BF), element_idx, xi)

        lines!(ax, x[:], form_eval[1], color=color_i, label=L"\phi_{%$basis_idx}")

        scatter!(ax, x[:][[1, end]], [0.0, 0.0], color=:tomato)
    end
end
fig[1, 2] = Legend(fig, ax, marge=true, unique=true)

fig = DisplayAs.Text(DisplayAs.PNG(fig))

# > **Note:** In both of the above examples, the only functions non-zero at ``x=0`` and
# > ``x=L`` are ``\phi_1`` and ``\phi_n``. This means that, in particular, the functions
# > ``\phi_2, \dots, \phi_{n-1}`` form a basis for ``W_n``. We will use this fact later on.
# Since, at each time instant ``t``, our approximate solution ``u_n(x,t)`` is represented
# as a linear combination of the basis functions ``\phi_i``, ``i = 1, \dots, n``, that
# span ``V_n``, this means that our approximate solution has the following form:
# ```math
# u_n(x,t) = \sum_{i=1}^n c_i(t) \phi_i(x)\;.
# ```
# In other words, the coefficients of the linear combination are time-dependent.
# But we can say more! Since ``u_n(0,t) = u_0`` and ``u_n(L,t) = u_L`` are the boundary
# conditions, then we must have:
# ```math
# u_n(x,t) = u_0\phi_1(x) + \sum_{i=2}^{n-1} c_i(t) \phi_i(x) + u_L\phi_n(x)\;.
# ```
# That is, the only unknown coefficients in the above expression are ``c_i(t)``,
# ``i = 2, \dots, n-1``.
#
# > **_ASSUMPTION:_** For simplicity, we assume that ``u_0`` and ``u_L`` are constants.

# ### B. Assembling the System of ODEs

# Now that we have arrived at an explicit form of our approximate solution to the weak
# problem, let us see how the discrete weak problem leads to a system of ODEs for the
# coefficients ``c_i(t)``. This process is called *assembly* and it leads to a system of
# ODEs that looks like:
# ```math
# \mathbf{M}\frac{d\mathbf{C}}{dt} + \mathbf{K} \mathbf{C} = \mathbf{F} - u_{0}\mathbf{F}^{b,0} - u_{L}\mathbf{F}^{b,L}\;,
# ```
# where we have arranged the *unknown* coefficients ``c_i(t)`` in a vector
# ``\mathbf{C}(t) := [c_2(t), c_3(t), \dots, c_{n-1}(t)]``.
#
# Some terminology: in the above system of ODEs, ``\mathbf{M}`` is called the mass matrix,
# `` \mathbf{K} `` is called the stiffness matrix, `` \mathbf{F} `` is called the load
# vector, ``\mathbf{F}^{b,0}`` and ``\mathbf{F}^{b,L}`` are the contributions of the known
# coefficients (``c_1`` and ``c_n``) to the loading, respectively, and `` \mathbf{C} `` is
# the vector of unknown coefficients that define the solution.
#
# The idea behind assembly is simple. We substitute the assumed form of our discrete
# solution ``u_n`` into the discrete weak problem. This gives us:
# ```math
# \int_{0}^{L} w_n\frac{\partial}{\partial t}(\sum_{j=1}^n c_j\phi_j)\;\mathrm{d}x + \int_{0}^{L} \alpha\frac{\partial w_n}{\partial x}\frac{\partial}{\partial x}(\sum_{j=1}^n c_j\phi_j) \,\mathrm{d}x = \int_0^L f w_n \, dx\,,\qquad \forall w_n\in W_n\;,
# ```
#
# Since we need to satisfy the above equation for all ``w_n \in W_n``, and since the above
# equation is linear in ``w_n``, it is actually enough if we satisfy the above equation for
# the basis functions that span ``W_n``, i.e., ``\phi_i``, ``i = 2, \dots, n-1``.
# Then, choosing ``w_n = \phi_i`` gives us the following equation, and we get one such
# equation for each ``i = 2, \dots, n-1``,
# ```math
# \int_{0}^{L} \phi_i\frac{\partial}{\partial t}(\sum_{j=1}^n c_j\phi_j)\;\mathrm{d}x + \int_{0}^{L} \alpha\frac{\partial \phi_i}{\partial x}\frac{\partial}{\partial x}(\sum_{j=1}^n c_j\phi_j) \,\mathrm{d}x = \int_0^L f \phi_i \, dx\;.
# ```
#
# We can rearrange this equation as:
# ```math
# \sum_{j=2}^{n-1}\frac{dc_j}{dt} \int_{0}^{L} \phi_i\phi_j\;\mathrm{d}x + \sum_{j=2}^{n-1}c_j\int_{0}^{L} \alpha\frac{d \phi_i}{d x}\frac{d \phi_j}{d x} \,\mathrm{d}x = \int_0^L f \phi_i - u_0\int_{0}^{L} \alpha\frac{d \phi_i}{d x}\frac{d \phi_1}{d x} \, dx - u_L\int_{0}^{L} \alpha\frac{d \phi_i}{d x}\frac{d \phi_n}{d x} \, dx\;.
# ```
#
# Then, it is easy to see that this equation represents the ODE system at the beginning of
# this section by defining:
# * ``\mathbf{M}_{ij} = \int_0^L \phi_i\phi_j\;\mathrm{d}x``,
# * ``\mathbf{K}_{ij} = \int_0^L \frac{d\phi_i}{dx}\frac{d\phi_j}{dx}\;\mathrm{d}x``,
# * ``\mathbf{F}_{i} = \int_0^L \phi_i f\;\mathrm{d}x``,
# * ``\mathbf{F}^{b,0}_{i} = \int_{0}^{L} \alpha\frac{d \phi_i}{d x}\frac{d \phi_1}{d x} \, dx``,
# * ``\mathbf{F}^{b,L}_{i} = \int_{0}^{L} \alpha\frac{d \phi_i}{d x}\frac{d \phi_n}{d x} \, dx\;.``
#
# To assemble the matrices ``\boldsymbol{\mathsf{M}}`` and ``\boldsymbol{\mathsf{K}}`` and
# the vectors ``F``, ``F^{b,0}`` and ``F^{b,L}``, we first must define ``\alpha`` and ``f``.
#
# Before choosing our forcing term, it is relevant to briefly analyse the behavior of our
# solution. We saw that our weak form of the equation is
# ```math
# \int_{0}^{L} w_n\frac{\partial u_n}{\partial t}\;\mathrm{d}x + \int_{0}^{L} \alpha \frac{\partial u_n}{\partial x}\frac{\partial w_n}{\partial x} \,\mathrm{d}x = \int_0^L f w_n \, dx\,,\qquad \forall w_n\in W_n\;,
# ```
#
# Since we have Dirichlet boundary conditions (i.e., we enforce the value of the
# temperature on both sides of our interval), if we prescribe a stationary heat source, the
# solution will evolve to a stationary state. This stationary state, ``u_{h}^{s}`` will be
# the one that satisfies
# ```math
# \int_{0}^{L} \alpha \frac{\partial u_n}{\partial x}\frac{\partial w_n}{\partial x} \,\mathrm{d}x = \int_0^L f w_n \, dx\,,\qquad \forall w_n\in W_n\;,
# ```
# or in matrix form
# ```math
# \mathbf{K} \mathbf{C} = \mathbf{F} - u_{0}\mathbf{F}^{b,0} - u_{L}\mathbf{F}^{b,L}\;.
# ```
#
# Additionally, if ``\alpha`` is constant, and if the heat source ``f`` is smooth, then we
# can easily construct analytical solutions to the stationary state. For example,
# ```math
# u^{s}(x, t) = 1 + \frac{1}{2}\cos\left(\frac{2\pi}{L} x\right)\,,
# ```
# if the heat source is
# ```math
# f(x, t) = \frac{2\alpha\pi^{2}}{L^{2}}\cos\left(\frac{2\pi}{L} x\right)\,.
# ```
# We choose a finite element space with ``(N,p,k) = (10,2,1)``, i.e., with more elements
#  compared to the last example. This is to ensure that we have sufficient accuracy for
# computing a decent solution.
