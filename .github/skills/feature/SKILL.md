---
name: Mantis Feature Development
description: Implement new functionality in the Mantis.jl finite element library while preserving FEEC abstractions, module boundaries, and project conventions.
---

# Mantis.jl Feature Development

This skill describes how to correctly implement new functionality in Mantis.jl.

## Core Principles

Mantis is organized as a layered FEEC library.

Every new feature should preserve the existing abstraction hierarchy.

Never bypass existing abstractions simply because something is easier to implement.

Whenever possible, implement functionality by extending existing interfaces instead of
introducing parallel implementations.

---

# Module Dependency Order

Modules have a strict dependency order.

```
GeneralHelpers
    ↓
Mesh
    ↓
Points
    ↓
Hierarchy
    ↓
Geometry
    ↓
FunctionSpaces
    ↓
Quadrature
    ↓
Forms
    ↓
Analysis
    ↓
Assemblers
    ↓
TimeIntegrators
    ↓
Plot
```

A module may depend on modules above it.

Avoid introducing dependencies in the opposite direction.

---

# Before Implementing

First identify which layer owns the feature.

Typical ownership:

- mesh topology → Mesh
- physical mappings → Geometry
- finite element spaces → FunctionSpaces
- symbolic FEEC operators → Forms
- numerical integration → Quadrature
- global matrix assembly → Assemblers
- postprocessing → Analysis
- visualization → Plot

Avoid implementing functionality in a higher layer when it naturally belongs in a lower one.

---

# Module Structure

Each module follows the same pattern.

```
ModuleName.jl

module ModuleName

exports

abstract types

fallback interface methods

include("...")

end
```

When adding new interfaces:

1. add the abstract interface
2. provide documented fallback methods throwing `MethodError`
3. implement concrete types in separate files
4. include those files from the module

Do not place large implementations directly inside `ModuleName.jl`.

---

# Geometry

Geometry maps

```
[0,1]^n
    →
physical space
```

Evaluation always occurs on the canonical domain.

Do not evaluate basis functions directly in physical coordinates.

Coordinate transforms belong inside Geometry.

Assemblers and Forms should remain geometry-independent.

---

# Forms

Forms are symbolic expression trees.

Operators such as

- d
- ★
- δ
- ∧
- ∫
- ♯

construct expressions.

They do **not** perform numerical evaluation.

Evaluation happens exclusively through `evaluate`.

When implementing a new operator:

- preserve lazy evaluation
- create new expression nodes
- extend `evaluate`
- avoid eager computation

---

# Function Spaces

New finite element spaces should extend `AbstractFESpace` and the corresponding documented
abstract methods.

Reuse existing interfaces whenever possible.

---

# Assemblers

Assemblers convert symbolic Forms into sparse matrices.

Weak formulations should build symbolic expressions.

Avoid inserting numerical kernels directly into weak formulations.

If new assembly logic is required:

- extend the assembler interface
- preserve sparse assembly
- reuse existing quadrature infrastructure

---

# Public API

When adding public functionality use `export` or `public` as appropriate

Keep exported APIs minimal.

---

# Documentation

Public APIs require docstrings.

Use the project style:

- Arguments (when not obvious from the name)
- Returns (when not obvious from the name)
- Examples (when useful)

Follow existing documentation style instead of inventing new conventions.

---

# Tests

Every feature should include tests.

Mirror the source layout.

Example:

```
src/Geometry/NewGeometry.jl

↓

test/Geometry/NewGeometryTests.jl
```

Register new test files in the corresponding

```
test/<Module>/runtests.jl
```

Test files contain top-level `@test`s.

Do not wrap them in additional `@testset`s.

---

# Extensions

Optional functionality belongs in package extensions.

Current example:

```
ext/MakieExt.jl
```

Do not introduce hard dependencies for optional features.

---

# Style

Follow BlueStyle with the repository's formatter configuration.

Match surrounding code.

Prefer extending existing APIs over introducing new naming patterns.

---

# Implementation Checklist

Before opening a PR verify that:

- feature lives in the correct module
- abstraction boundaries are preserved
- module dependency order is respected
- lazy evaluation is preserved (Forms)
- new public APIs are documented
- exports are updated
- tests are added
- module `runtests.jl` registers the new tests
- Julia 1.10 compatibility is preserved
- optional dependencies use package extensions

---

# Things to Avoid

Do not

- evaluate symbolic Forms eagerly
- couple Assemblers to specific form or finite element spaces
- introduce circular module dependencies
- perform geometry computations outside Geometry
- duplicate existing interfaces
- add optional packages as hard dependencies
