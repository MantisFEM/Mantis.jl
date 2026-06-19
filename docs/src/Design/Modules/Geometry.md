```@meta
CurrentModule = Mantis.Geometry
```
# [Geometry](@id DocGeometryModule)

Mantis' `Geometry` module describes the *domain* a problem is posed on. A geometry is a
collection of mappings from the canonical element ``[0, 1]^n`` into physical space, and every
`evaluate` call elsewhere in `Mantis` (on a function space, a form, a quadrature rule) is
ultimately referred back to one of these mappings. The formal definition is given below, after
a short tour of the geometry types.

## Geometry types

All geometries are subtypes of [`AbstractGeometry`](@ref), parametrised by the manifold
dimension `n`, the image (embedding) dimension `m`, and the number of patches. The concrete
types fall into a few groups:

- **Base geometries** describe a domain directly.
  [`CartesianGeometry`](@ref) is an axis-aligned box mesh defined by per-direction
  breakpoints, and is the usual starting point. [`UnstructuredGeometry`](@ref) and
  [`DiscreteGeometry`](@ref) describe more general meshes.
- **Tensor-product geometries** ([`TensorProductGeometry`](@ref)) build a higher-dimensional
  geometry from lower-dimensional factors; the construction and its Jacobian are derived in
  detail [below](@ref GeometryTensorProduct).
- **Derived geometries** wrap another geometry to transform it.
  `MappedGeometry` composes a base geometry with a user-supplied `Mapping` to produce curved
  domains (see the [One-dimensional mapped geometry](@ref) example);
  [`HierarchicalGeometry`](@ref) carries the active-element structure needed for adaptive
  refinement; and [`MaskedGeometry`](@ref) restricts a geometry to a subset of its elements.

A geometry is created either directly or through a convenience helper. The most common is
[`create_cartesian_box`](@ref):

```julia
using Mantis

# A unit square with 4×4 elements:
geometry = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (4, 4))

# A curved version of the same domain:
mapping     = Geometry.create_curvilinear_mapping((0.0, 0.0), (1.0, 1.0))
curved_geo  = Geometry.MappedGeometry(geometry, mapping)
```

## Evaluation, Jacobian and Hessian

Because integrals are pulled back to the canonical element, `Mantis` needs not only the
mapping but also its derivatives. Every geometry therefore supports three evaluation routines,
each taking an `element_idx` and a set of [Points](@ref):

- [`evaluate`](@ref) returns the physical coordinates ``\Phi_i(\xi)``;
- [`jacobian`](@ref) returns the first derivatives ``\partial \Phi_i / \partial \xi``;
- [`hessian`](@ref) returns the second derivatives, needed for higher-order operators such as
  the Laplacian in the [Biharmonic](@ref) example.

The metric tensor ``g`` and its determinant (see `metric`/`inv_metric` below) are assembled
from the Jacobian and are what make metric-dependent operators such as the
[Hodge star](@ref FormsHodge) and [codifferential](@ref FormsCodifferential) work on curved
geometries.

## [Geometry: formal definition](@id GeometryDefinition)

An ``(n, m)`` geometry ``\Phi`` is a collection of ``L`` mappings
``\left\{\Phi_{i}\right\}_{i=1}^{L}`` that map the canonical ``n``-dimensional domain,
``\Omega^{0} := [0, 1]^{n}`` into ``L`` ``n``-dimensional simply connected subdomains,
``\Omega^{1}_{i}`` with ``i = 1, \dots, L``, of ``\mathbb{R}^{m}``. Moreover,
``\bigcap_{i=1}^{L}\Omega^{1}_{i} = \emptyset`` and
``\overline{\Omega}^{1}_{i} \cap \overline{\Omega}^{1}_{j} \subset \partial\Omega^{1}_{i}
\cup \partial\Omega^{1}_{j}`` with ``i,j = 1, \dots, L``.

Note that
```math
\Phi_{i}(\xi_{1}, \dots, \xi_{n}) = (x_{1}, \dots, x_{m}),
```

and we use ``\Phi_{i, j} = x_{j}`` to mean the ``j``-th component of the mapping ``\Phi`` of
element ``i``.

## [Tensor Product Geometry](@id GeometryTensorProduct)

Given an ``(n_{1}, m_{1})`` geometry ``\Phi^{1}`` of ``L_{1}`` mappings and an ``(n_{2},
m_{2})`` geometry ``\Phi^{2}`` of ``L_{2}`` mappings, i.e.,
```math
\Phi^{1}_{i}: [0, 1]^{n_{1}} \mapsto \Omega^{1}_{i} \subset \mathbb{R}^{m_{1}}, \quad i = 1,
\dots, L_{1}
```
and
```math
\Phi^{2}_{i}: [0, 1]^{n_{2}} \mapsto \Omega^{2}_{i} \subset \mathbb{R}^{m_{2}}, \quad i = 1,
\dots, L_{2}
```
the tensor product geometry ``\Phi := \Phi^{1}\otimes\Phi^{2}`` is an ``(n_{1} + n_{2},
m_{1} + m_{2})`` geometry made up of a collection of ``L_{1}L_{2}`` mappings ``\Phi_{k}``
```math
\Phi_{k = L_1(j - 1) + I}: [0, 1]^{n_{1}} \times [0, 1]^{n_{2}} \mapsto \Omega_{k} =
\Omega^{1}_{i}\times\Omega^{2}_{j} \subset \mathbb{R}^{m_{1} + m_{2}}, \quad i = 1, \dots,
L_{1}, \text{ and } j = 1, \dots, L{2}.
```
Specifically, we have
```math
\Phi_{L_1(j - 1) + i, l}(\xi_{1}, \dots, \xi_{n}) := \left\{ \begin{array}{ll} \Phi^{1}_{i,
l}(\xi_{1}, \dots, \xi_{n_{1}}), & \quad \text{if } l \leq n_{1}\\ \Phi^{2}_{j, l -
n_{1}}(\xi_{n_{1} + 1}, \dots, \xi_{n_{1} + n_{2}}), & \quad \text{if } n_{1} < l \leq n_{1}
+ n_{2} \end{array} \right.\,, \quad i = 1, \dots, L_{1},\quad j = 1, \dots, L_{2}, \text{
  and } l = 1, \dots, m_{1} + m_{2}\,.
```

The Jacobian of this geometry
```math
J^{k}_{l,v} := \frac{\partial \Phi_{k, l}}{\partial \xi_{v}}
```
is given by
```math
\frac{\partial\Phi_{L_1(j - 1) + i, l}}{\partial\xi_{v}}(\xi_{1}, \dots, \xi_{n}) := \left\{
\begin{array}{ll} \frac{\partial\Phi^{1}_{i, l}}{\partial \xi_{v}}(\xi_{1}, \dots,
\xi_{n_{1}}), & \quad \text{if } l \leq n_{1}, \text{  and  } v \leq m_{1}\\
\frac{\partial\Phi^{2}_{j, l - n_{1}}}{\partial \xi_{u - m_{1}}}(\xi_{n_{1} + 1}, \dots,
\xi_{n_{1} + n_{2}}), & \quad \text{if } n_{1} < l \leq n_{1} + n_{2}, \text{  and  } m_{1}
< v \leq m_{1} + m_{2} \\ 0 & \quad\text{otherwise} \end{array} \right.\,, \quad i = 1,
\dots, L_{1}, \quad j = 1, \dots, L_{2}, \text{ and } l = 1, \dots, m_{1} + m_{2}\,.
```

### Evaluation

Given the `NTuple` `ξ` of ``n`` `Vectors`, ``\boldsymbol{\xi}^{i}``, ``i=1, \dots, n``, each
containing ``m_{i}`` unidimensional coordinates ``\xi^{i}_{j}``, ``i = 1, \dots, n`` and ``
j = 1, \dots m_{i}``, the tensor product geometry is evaluated at the element `element_idx`
and at the ``\prod_{i=1}^{n}m_{i}`` tensor product points ``V_{k = j_{1} + \sum_{i=2}^{n}
(j_{i} - 1)\prod_{l=1}^{i-1}m_{l}} = (\xi^{1}_{j_{1}}, \dots, \xi^{n}_{j_{n}})``, with
``j_{i} = 1, \dots, m_{i}``.

The output is a matrix, ``\boldsymbol{\mathsf{X}}`` of dimensions
``\left(\prod_{i=1}^{n}m_{i}\right) \times m`` (the number of tensor product points where
the geometry is evaluated in element `element_idx`, and the dimension of the embedding space
to where the canonical element is mapped into. Specifically:
```math
\boldsymbol{\mathsf{X}}_{k, l} = \Phi_{r, l}(\xi^{1}_{j_{1}}, \dots,
\xi^{n}_{j_{n}}),
```
where ``r =`` `element_idx`, and ``k = j_{1} + \sum_{i=2}^{n} (j_{i} -
1)\prod_{l=1}^{i-1}m_{l}``, as before.

### Jacobian

Given the `NTuple` `ξ` of ``n`` `Vectors`, ``\boldsymbol{\xi}^{i}``, ``i=1, \dots, n``, each
containing ``m_{i}`` unidimensional coordinates ``\xi^{i}_{j}``, ``i = 1, \dots, n`` and ``
j = 1, \dots m_{i}``, evaluates the Jacobian of the tensor product geometry at the element
`element_idx` and at the ``\prod_{i=1}^{n}m_{i}`` tensor product points ``V_{k = j_{1} +
\sum_{i=2}^{n} (j_{i} - 1)\prod_{l=1}^{i-1}m_{l}} = (\xi^{1}_{j_{1}}, \dots,
\xi^{n}_{j_{n}})``, with ``j_{i} = 1, \dots, m_{i}``.

The output is a matrix, ``\boldsymbol{\mathsf{J}}`` of dimensions
``\left(\prod_{i=1}^{n}m_{i}\right) \times m \times n`` (the number of tensor product points
where the geometry is evaluated in element `element_idx`, the dimension of the embedding
space to where the canonical element is mapped into, and the dimension of the canonical
element, which is the same as the dimension of the element's manifold). Specifically:
```math
\boldsymbol{\mathsf{J}}_{k, l, s} = \frac{\partial\Phi_{r,
l}}{\partial\xi_{s}}(\xi^{1}_{j_{1}}, \dots, \xi^{n}_{j_{n}}),
```
where ``r = \mathtt{element\_idx}``, and ``k = j_{1} + \sum_{i=2}^{n} (j_{i} -
1)\prod_{l=1}^{i-1}m_{l}``, as before.


## All docstrings from Mantis.Geometry
```@autodocs
Modules = [Main.Mantis.Geometry]
```
