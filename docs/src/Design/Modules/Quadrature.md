```@meta
CurrentModule = Mantis.Quadrature
```
# [Quadrature](@id DocQuadratureModule)

```@docs
Quadrature
```

## Overview

Every integral that `Mantis` computes — mass and stiffness matrices, load vectors, error
norms — is evaluated numerically. A quadrature rule replaces an integral by a weighted sum
over a finite set of nodes,
```math
\int_{\Omega^{0}} f(\xi) \, \mathrm{d}\xi \approx \sum_{i=1}^{N} w_{i} \, f(\xi_{i}),
```
where ``\xi_{i}`` are the *nodes* and ``w_{i}`` the *weights*. The `Quadrature` module
supplies these pairs and the bookkeeping needed to apply them across a mesh.

Two conventions shape the whole module. First, rules live on the canonical domain
``\Omega^{0} := [0, 1]^{n}``, where ``n`` is the manifold dimension, rather than on the
``[-1, 1]`` interval of the classical literature. This matches the convention that
[Points](@ref DocPointsModule) and [Geometry](@ref DocGeometryModule) already use, so one
rule can be reused on every element of a mesh without remapping; the geometry supplies the
Jacobian factor that accounts for the element's actual size. Consequently the weights of a
rule sum to ``1``, the measure of ``\Omega^{0}``, rather than to ``2``.

Second, nodes are stored as [Points](@ref DocPointsModule) objects rather than as bare
arrays. A rule is therefore a valid argument anywhere in `Mantis` that accepts evaluation
points, and multidimensional rules retain the tensor-product structure that function-space
and geometry evaluation exploit.

## [Element and global rules](@id QuadratureHierarchy)

Integration in a finite element method happens element by element, but the caller usually
wants to integrate over a whole domain. The module keeps these two levels apart. Both derive
from a common root carrying only the manifold dimension:

```@docs
AbstractQuadratureRule
```

An *element* rule is a concrete set of nodes and weights, valid on one canonical element:

```@docs
AbstractElementQuadratureRule
```

A *global* rule covers an entire domain, and answers questions about which element rule
applies where:

```@docs
AbstractGlobalQuadratureRule
```

Separating the two means a single element rule can be shared by every element of a mesh
without being copied, while the global rule remains free to vary the rule from element to
element — which is what non-uniform or hierarchically refined meshes need.

## [Element quadrature rules](@id QuadratureElementRules)

Element rules are represented by a single concrete type:

```@docs
CanonicalQuadratureRule
```

Every element rule exposes its nodes, its weights, and a human-readable label recording how
it was built:

```@docs
get_nodes
get_weights
get_label
```

The label exists for verification rather than for dispatch. Because rules are composed
(a tensor-product rule reports the rules it was built from), a mislabelled or unintended rule
is visible on inspection instead of silently changing the accuracy of a computation.

### [One-dimensional rules](@id QuadratureRules1D)

The module builds every rule from one-dimensional constructors, each returning a
[`CanonicalQuadratureRule{1}`](@ref CanonicalQuadratureRule). They differ in where they place
nodes, and therefore in accuracy and in whether the endpoints of the interval are included:

```@docs
gauss_legendre
gauss_lobatto
```

Gauss-Legendre places all nodes strictly inside the interval and is exact for polynomials of
degree ``2N - 1`` with ``N`` nodes, the best attainable. Gauss-Lobatto sacrifices two degrees
of exactness in order to include ``\xi = 0`` and ``\xi = 1``, which matters when the
integrand must be sampled on element boundaries.

```@docs
clenshaw_curtis
newton_cotes
```

Clenshaw-Curtis uses Chebyshev nodes, also including the endpoints, and although its
guaranteed degree of exactness is only ``p``, in practice it often performs comparably to
Gauss quadrature. Newton-Cotes uses equally spaced nodes, in a *closed* variant including the
endpoints and an *open* variant excluding them; equally spaced nodes make it the natural
choice when the integrand is only available on a uniform grid, at the cost of poor
conditioning for large numbers of points.

### [Tensor-product rules](@id QuadratureTensorProduct)

Multidimensional rules are formed from one-dimensional factors, taking the Cartesian product
of the nodes and the outer product of the weights:

```@docs
tensor_product_rule
```

The resulting nodes are a [`Points.TensorProductPoints`](@ref), so downstream evaluation can
recover the one-dimensional factors and exploit them, rather than treating the rule as an
unstructured cloud of points. This is the reason the module stores nodes as points objects at
all: on a ``2 \times 2 \times 2`` rule the distinction is irrelevant, but tensor-product
function spaces evaluated at tensor-product nodes are what make higher-degree computations
affordable.

## [Global quadrature rules](@id QuadratureGlobalRules)

A global rule maps elements of a mesh to the element rules that integrate them. Two notions
of "element" appear here, and the distinction matters when reading the rest of `Mantis`:

- a **base element** is an element of the underlying mesh;
- a **quadrature element**, also called an *evaluation element*, is a smallest unit on which
  quadrature is actually performed.

They coincide in the simple case, but a single base element may be subdivided into several
quadrature elements — for instance when an integrand is only piecewise smooth across a base
element, so that integrating it in one piece would lose accuracy. Any global rule reports
both counts, and resolves a base element into its quadrature elements:

```@docs
get_num_evaluation_elements
get_num_base_elements
get_element_idxs
```

Given a quadrature element, the rule then supplies the element rule to use on it:

```@docs
get_element_quadrature_rule
```

These four functions are the whole interface a global rule must provide; the default
implementations of the last two exist only to produce a clear error for types that have not
implemented them.

### [StandardQuadrature](@id QuadratureStandard)

The common case is the same rule everywhere, with base and quadrature elements coinciding:

```@docs
StandardQuadrature
get_canonical_quadrature_rule
```

This is what most of `Mantis` uses. Constructing one requires only an element rule and the
number of elements in the geometry:

```@repl QuadratureExample
using Mantis
qrule = Quadrature.tensor_product_rule((3, 3), Quadrature.gauss_legendre);
Quadrature.get_label(qrule)
Points.get_input_points(Quadrature.get_nodes(qrule))
sum(Quadrature.get_weights(qrule))
dΩ = Quadrature.StandardQuadrature(qrule, 4);
Quadrature.get_num_evaluation_elements(dΩ)
```

## [Helper functions](@id QuadratureHelpers)

Discretisations built on a de Rham complex frequently need several rules at once, of the same
family but with different numbers of points. Building them individually is verbose, so the
module provides a helper that constructs a whole tuple:

```@docs
get_canonical_quadrature_rules
```

A companion function, `get_global_quadrature_rules`, wraps each of these in a
[`StandardQuadrature`](@ref) for a given number of elements.

### [Internals: tensor-product weights](@id QuadratureInternalWeights)
!!! note "Internal behaviour"
    We explain how tensor-product weights are formed. However, this is considered an
    implementational detail.

The weights of a tensor-product rule are the products of the factor weights, laid out in the
linear order that [TensorProducts](@ref DocTensorProductsModule) uses for the corresponding
nodes, so that weight `i` always pairs with node `i`.

```@docs
_compute_tensor_product
```

## [Relationships to Other Modules](@id QuadratureRelationships)

- **[Points](@ref DocPointsModule)**: quadrature nodes *are* `AbstractPoints`, and
  multidimensional rules use `TensorProductPoints`.
- [TensorProducts](@ref DocTensorProductsModule): supplies the index bookkeeping that relates
  the linear ordering of tensor-product nodes and weights to their per-dimension factors.
- **[Forms](@ref)**: the [integral operator](@ref FormsIntegrals) stores an
  `AbstractGlobalQuadratureRule` and uses it to evaluate integrals of forms; the terminology
  of evaluation elements used there is the one defined above.
- [Assemblers](@ref DocAssemblyModule): weak formulations take global rules as arguments and
  pass them to the integrals they assemble.
- [Geometry](@ref DocGeometryModule): supplies the Jacobian that maps an integral on the
  canonical domain to the physical element, which is why the rules themselves need no
  knowledge of the mesh.

## [References](@id QuadratureReferences)

The Clenshaw-Curtis implementation follows the FFT-based algorithm of [Waldvogel2006](@cite);
for its accuracy relative to Gauss quadrature see [Trefethen2008](@cite) and
[Trefethen2022](@cite).
