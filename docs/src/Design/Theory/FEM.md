# The Finite Element Method (FEM)

FEM is a versatile numerical method for finding approximate solutions to partial differential equations (PDEs).
In this section, we introduce the fundamental concepts of FEM through a classical, second-order elliptic PDE: the Poisson equation.

## The model problem: Poisson equation

Consider a bounded domain $\Omega \subset \mathbb{R}^k$, $k \in \{1, 2, 3\}$, with a sufficiently smooth boundary $\partial\Omega$. We seek a function $u$ defined on $\Omega$ that satisfies the Poisson equation with homogeneous Dirichlet boundary conditions:

```math
\begin{align*}
-\Delta u &= f \quad \text{in } \Omega, \\
u &= 0 \quad \text{on } \partial\Omega,
\end{align*}
```
where $f$ is a given source function, and $\Delta = \nabla \cdot \nabla$ is the Laplace operator.

## Strong form vs. weak form

The equation above represents the **strong form** of the problem.
It requires the solution $u$ to have continuous second derivatives, which is often a very restrictive condition, especially when $f$ is not smooth.

To relax this regularity requirement, we derive the **weak (or variational) form**.
We multiply the strong form by an arbitrary test function $v$ that vanishes on the boundary ($v=0$ on $\partial\Omega$) and integrate over the domain $\Omega$:

```math
-\int_\Omega (\Delta u) v \, dx = \int_\Omega f v \, dx.
```

Applying integration by parts (Green's first identity) to the left-hand side, we get:

```math
\int_\Omega \nabla u \cdot \nabla v \, dx - \int_{\partial\Omega} (\nabla u \cdot n) v \, ds = \int_\Omega f v \, dx.
```

Since the test function $v$ vanishes on the boundary $\partial\Omega$, the boundary integral is zero.
The resulting problem can then be stated in the following form: find $u \in V$ such that

```math
a(u, v) = l(v) \quad \forall v \in V,
```

where
*   $V = H^1_0(\Omega)$ is the Sobolev space of functions with square-integrable weak derivatives that vanish on the boundary;
*   $a(u, v) = \int_\Omega \nabla u \cdot \nabla v \, dx$ is a symmetric bilinear form;
*   $l(v) = \int_\Omega f v \, dx$ is a linear form.

This formulation requires $u$ and $v$ to only have (square-integrable) first derivatives and thus describes a much larger class of situations than the strong form.

## Bubnov-Galerkin approximation

The weak formulation is infinite-dimensional because the space $V$ is infinite-dimensional.
The **(Bubnov-)Galerkin method** consists of replacing the continuous space $V$ with a finite-dimensional subspace $V_h \subset V$.
(The subscript here contains a positive parameter $h$ which characterises the discrete nature of the problem, with $h \rightarrow 0$ denoting that we are approaching the infinite-dimensional setting.)

The corresponding discrete problem can then be written down as: find $u_h \in V_h$ such that

```math
a(u_h, v_h) = l(v_h) \quad \forall v_h \in V_h.
```

To compute the solution, we will take $V_h$ to be a finite element space consisting of piecewise-defined functions on a suitable partition of the domain called a mesh.

For instance, in 1D, an $m$-element mesh for $\Omega = (a, b)$ can be defined by choosing mesh vertices $x_i$, $i = 0, \dots, m$:

```math
a = x_0 < x_1 < \cdots < x_m = b.
```

The $i$-th mesh element is defined to be $\Omega_i = (x_{i-1}, x_{i})$, $i = 1, \dots, m$.
Finally, an example finite element space on this mesh could be the space of $C^1$-smooth piecewise-quadratic functions satisfying homogeneous boundary conditions:

```math
V_h := \big\{ f \in C^1_0(\Omega) : f|_{\Omega_i} \in \mathcal{P}_2~,~i = 1, \dots, m \big\}.
```

The parameter $h$ characterizing the discrete nature of this space could be defined in different ways; for instance, a couple of popular choices are $h = 1/\dim(V_h)$ and $h = \max_i (x_{i} - x_{i-1})$.

In the general setting, once we have chosen such a finite element space, let $n = \dim(V_h)$ and let $\{\phi_1, \phi_2, \dots, \phi_n\}$ be a basis for $V_h$.
Then, we can express the approximate solution $u_h$ as a linear combination of these basis functions:

```math
u_h = \sum_{j=1}^n u_j \phi_j,
```

where $u_j \in \mathbb{R}$ are the unknown coefficients called degrees of freedom.

Testing the discrete problem with each basis function $v_h = \phi_i$ for $i=1, \dots, n$, we obtain a system of linear equations:

```math
\sum_{j=1}^n a(\phi_j, \phi_i) u_j = l(\phi_i) \quad \text{for } i=1,\dots,n.
```

This can be written in matrix-vector form as:

```math
K U = F,
```

where
*   $K$ is the matrix with entries $K_{ij} = a(\phi_j, \phi_i) = \int_\Omega \nabla \phi_j \cdot \nabla \phi_i \, dx$.
*   $F$ is the source vector with entries $F_i = l(\phi_i) = \int_\Omega f \phi_i \, dx$.
*   $U = [u_1, u_2, \dots, u_n]^T$ is the vector of unknowns.
