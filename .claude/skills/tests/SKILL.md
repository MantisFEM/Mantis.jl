---
name: Mantis Testing
description: Write and maintain tests for Mantis.jl that verify correctness, numerical accuracy, API behavior, and regression safety while following the project's testing conventions.
---

# Mantis.jl Testing

This skill describes how to write tests for Mantis.jl.

Tests should verify **observable behaviour**, not implementation details.

Prefer testing through the public interface whenever possible.

---

# Testing Philosophy

A good test should:

- verify mathematical correctness
- verify numerical correctness
- catch regressions
- remain readable
- be deterministic
- be independent of execution order

Avoid tests that depend on implementation details unless those details are part of a
documented interface.

---

# Test Organization

Mirror the source tree.

Example:

```
src/
    Geometry/
        TensorProductGeometry.jl

↓

test/
    Geometry/
        TensorProductGeometryTests.jl
```

Every new test file must be registered inside the module's

```
test/<Module>/runtests.jl
```

which is then included from

```
test/runtests.jl
```

Do **not** wrap individual test files inside additional `@testset`s.

Each test file should contain top-level `@test`s, as the enclosing `@testset` is provided by
the module's `runtests.jl`.

---

# What to Test

When adding new functionality, consider testing:

## Public API

Verify that public functions:

- accept expected inputs
- return expected outputs
- throw expected exceptions
- preserve documented behaviour

---

## Mathematical Properties

Whenever possible, test mathematical identities rather than individual implementation steps.

Examples include:

- exact polynomial reproduction
- commuting diagrams
- exactness of complexes
- partition of unity
- interpolation properties
- symmetry
- conservation laws

---

## Numerical Accuracy

For numerical algorithms, compare against:

- analytical solutions
- manufactured solutions
- reference values
- previously validated implementations

Use tolerances appropriate for floating-point arithmetic.

Avoid exact equality unless mathematically guaranteed.

---

## Edge Cases

Consider testing:

- empty inputs
- degenerate meshes
- boundary elements
- single-element meshes
- high polynomial orders
- low polynomial orders
- invalid arguments

Ensure errors are informative and intentional.

---

## Regression Tests

When fixing a bug:

1. write a test that reproduces the bug
2. verify it fails before the fix
3. verify it passes afterwards

Keep the regression test even after the bug is fixed.

---

# Forms

Since Forms are symbolic expression trees, test both:

- symbolic construction
- numerical evaluation

For example, verify that operators construct the expected expressions and that `evaluate`
produces the correct numerical values.

Avoid testing internal tree layouts unless they are part of the public API.

---

# Geometry

Geometry tests should verify:

- mappings from canonical to physical coordinates
- Jacobians
- inverse mappings (when applicable)
- orientations
- boundary mappings

Prefer analytical geometries with known exact results.

---

# Function Spaces

Test:

- basis evaluation
- interpolation
- continuity
- partition of unity
- derivatives
- degree-specific behaviour

Whenever possible, compare against exact polynomial values.

---

# Quadrature

Verify:

- exact integration of polynomials or other spaces up to the expected degree
- correct handling of weights
- correct handling of mapped geometries

---

# Assemblers

Assembly tests should verify:

- correct matrix sizes
- sparsity structure
- symmetry (when expected)
- consistency with analytical weak forms

For larger systems, compare assembled operators against known reference solutions or
invariants rather than individual entries.

---

# Floating-Point Comparisons

Use tolerances appropriate to the computation.

Prefer

```
@test isapprox(...)
```

over exact equality for floating-point results.

Choose tolerances based on expected numerical error rather than arbitrary values.

---

# Determinism

Tests should produce identical results across runs.

Avoid dependence on:

- random number generation (unless seeded)
- execution order
- iteration order of unordered collections
- machine-specific behaviour

---

# Performance

Performance tests generally do not belong in the standard test suite.

If verifying a performance improvement:

- use dedicated benchmarks
- measure allocations when appropriate
- avoid fragile timing-based assertions

---

# Inference Tests

JET inference tests live under

```
test/Inference
```

They use a separate project and are only executed on supported Julia versions.

Do not place ordinary correctness tests there.

---

# Documentation Examples

If adding a `jldoctest` example:

- ensure it executes successfully
- keep it minimal
- avoid unnecessary complexity

Documentation examples should remain synchronized with the implementation.

---

# Test Checklist

Before opening a PR, verify that:

- new functionality has corresponding tests
- tests mirror the source tree
- new test files are registered in `runtests.jl`
- public APIs are tested
- edge cases are covered where appropriate
- floating-point comparisons use appropriate tolerances
- regression tests are added for bug fixes
- tests are deterministic
- the full test suite passes on supported Julia versions

---

# Things to Avoid

Do not:

- test private implementation details unnecessarily
- duplicate existing tests without increasing coverage
- rely on execution order
- hard-code floating-point values without justification
- use unnecessarily loose tolerances
- remove regression tests after a bug is fixed
- write brittle tests that fail after harmless refactoring

Prefer tests that validate mathematical correctness and user-visible behaviour over tests
that mirror the implementation.
