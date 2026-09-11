---
name: Mantis Refactoring
description: Refactor Mantis.jl code while preserving FEEC abstractions, module boundaries, API stability, and numerical correctness.
---

# Mantis.jl Refactoring

This skill describes how to safely refactor existing code in Mantis.jl.

The primary objective of a refactor is to improve the implementation **without changing
observable behaviour**.

Observable behaviour includes:

- numerical results
- public APIs
- supported Julia versions
- performance characteristics (unless intentionally improved)
- documentation semantics

When in doubt, prefer smaller, incremental refactors over large rewrites.

---

# Understand Before Changing

Before modifying code, identify:

- which module owns the functionality
- whether the code is part of the public API
- whether it implements an interface used elsewhere
- whether it is performance critical

Read the module's top-level `<Module>.jl` file first to understand the intended abstractions
before changing implementation files.

Never refactor code that you do not yet understand.

---

# Preserve Module Boundaries

Mantis follows a layered architecture.

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

Do not introduce dependencies that violate this ordering.

If functionality naturally belongs in a lower layer, move it there rather than duplicating
it in higher layers.

---

# Preserve Existing Abstractions

Prefer strengthening existing abstractions over introducing new ones.

When similar implementations exist:

- extract common functionality
- reuse existing interfaces
- eliminate duplication

Avoid introducing parallel interfaces that solve the same problem.

If multiple concrete types implement nearly identical logic, consider moving shared
behaviour into common helper functions or abstract interface methods.

---

# Preserve FEEC Semantics

The FEEC abstractions are fundamental to Mantis.

In particular:

- Forms represent symbolic expression trees.
- Operators build expressions.
- Evaluation occurs through `evaluate`.
- Assemblers consume symbolic expressions.
- Geometry handles mappings between canonical and physical domains.

Do not blur these responsibilities during refactoring.

Avoid introducing shortcuts that bypass symbolic representations.

---

# Preserve Public APIs

Refactoring should not require downstream users to modify their code.

Avoid:

- renaming exported symbols
- changing method signatures
- changing return types
- changing documented behaviour

If a breaking change is unavoidable, isolate it into a dedicated change rather than
combining it with unrelated refactoring.

---

# Reduce Duplication Carefully

When removing duplicated code:

- verify that implementations are truly equivalent
- preserve specialized behaviour
- avoid over-generalization

A small amount of duplication is preferable to an abstraction that obscures the mathematics
or FEEC concepts.

---

# Improve Readability

Refactoring should make code easier to understand.

Common improvements include:

- extracting helper functions
- improving function names
- reducing nesting
- separating independent responsibilities
- simplifying dispatch

Avoid introducing additional layers of abstraction unless they clearly improve
maintainability.

---

# Preserve Performance

Many parts of Mantis execute inside assembly loops.

When refactoring performance-critical code:

- avoid unnecessary allocations
- preserve type stability
- avoid introducing dynamic dispatch
- keep hot paths simple

Prefer measuring performance before and after significant refactors.

Do not assume cleaner code is automatically faster.

---

# Preserve Generic Programming

Mantis relies heavily on Julia's multiple dispatch.

Prefer extending existing generic interfaces rather than introducing type checks or
conditional logic.

Avoid replacing dispatch with manual branching based on concrete types.

---

# Documentation

If public code is moved or reorganized:

- keep docstrings attached to the public API
- update examples if needed
- ensure exported symbols remain documented

Internal helper functions generally do not require extensive documentation unless they
implement non-trivial algorithms.

---

# Tests

Every refactor should preserve existing test coverage.

If behaviour is unchanged:

- existing tests should continue to pass without modification

If code becomes easier to test:

- consider adding focused regression tests
- preserve mirror structure between `src/` and `test/`

Never remove tests solely because the implementation changed.

---

# Optional Dependencies

Do not convert weak dependencies into hard dependencies.

Functionality related to optional packages should remain inside package extensions under
`ext/`.

---

# Refactoring Checklist

Before opening a PR, verify that:

- no public APIs changed unintentionally
- module dependency order is preserved
- FEEC abstractions remain intact
- symbolic evaluation remains lazy
- geometry responsibilities remain in Geometry
- duplicated logic has been reduced appropriately
- performance-critical code remains type-stable
- existing tests pass
- new regression tests are added where appropriate
- documentation remains accurate

---

# Things to Avoid

Do not:

- rewrite working code without a clear objective
- introduce circular module dependencies
- replace dispatch with manual type checks
- move geometry logic into Assemblers or Forms
- eagerly evaluate symbolic expressions
- over-generalize small pieces of duplicated code
- combine behavioural changes with large refactors
- sacrifice numerical clarity for clever abstractions

Prefer refactors that are incremental, well-scoped, and easy to review.
