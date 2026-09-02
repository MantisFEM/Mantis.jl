---
name: Mantis Docstrings
description: Write clear, consistent, and informative docstrings for Mantis.jl that follow the project's documentation style and accurately describe the public API.
---

# Mantis.jl Docstrings

This skill describes how to write docstrings for Mantis.jl.

Public APIs should be documented clearly enough that users can understand how to use them
without reading the implementation.

Docstrings are part of the public interface and should evolve alongside the code.

---

# What Should Be Documented

Document all public:

- functions
- methods with user-facing behavior
- types
- macros
- constants when appropriate

Internal helper functions generally do not require docstrings unless they implement
important algorithms or are likely to be reused.

---

# Follow Existing Style

Match the style already used throughout `src/`.

Do not introduce a new documentation format.

Public docstrings should use section headings where appropriate:

- `# Arguments` (when not obvious from the name)
- `# Returns` (when not obvious from the name)
- `# Exceptions`
- `# Examples` (whenever possible)

Only include sections that add value. Avoid empty or trivial sections.

---

# Function Docstrings

Begin with a concise summary describing what the function does.

Prefer describing the operation rather than the implementation.

Prefer:

> Compute the pullback of a differential form onto the reference domain.

Avoid:

> This function loops over the basis functions and applies pullback transformations.

Implementation details belong in comments, not the public documentation.

---

# Arguments

Include an `# Arguments` section whenever the function accepts non-obvious parameters.

For each argument, describe:

- its purpose
- expected type or interface (when not obvious)
- any assumptions or restrictions

Explain the _meaning_ of an argument, not merely its Julia type.

Prever:

```
- `geometry`: Geometry used to map the canonical element to physical space.
```

Avoid:

```
- `geometry`: An `AbstractGeometry`.
```

---

# Returns

Include a `# Returns` section whenever the return value is not immediately obvious.

Describe:

- what is returned
- important guarantees
- ownership or mutability when relevant

Focus on the semantic meaning of the result rather than its exact implementation type unless
that type is important to users.

---

# Exceptions

Include a `# Exceptions` section when the function intentionally throws errors under
documented conditions.

Describe:

- when an exception occurs
- why it occurs

---

# Examples

Include `# Examples` when they improve understanding (should be often).

Use `jldoctest` blocks.

Examples should:

- be minimal
- be complete
- execute successfully
- demonstrate typical usage

Prefer realistic examples over artificial ones.

---

# Type Docstrings

Describe:

- what the type represents
- its role within Mantis
- important invariants
- when users should use it

Avoid documenting every field unless users are expected to construct the type directly.

Explain the abstraction before the representation.

---

# Mathematical Concepts

Mantis is based on FEEC and differential geometry.

When documenting mathematical objects:

- use correct mathematical terminology
- define symbols when necessary
- distinguish between canonical and physical domains
- distinguish symbolic expressions from evaluated quantities

Assume readers have some numerical PDE background, but avoid unnecessary jargon.

---

# Generic Interfaces

Abstract interfaces should document:

- the purpose of the interface
- required methods
- expected behavior
- semantic contracts

Concrete implementations should document only behavior that differs from or extends the
interface documentation.

Avoid duplicating documentation across implementations.

---

# Keep Documentation Focused

Describe:

- what the API does
- why it exists
- when it should be used

Avoid lengthy implementation discussions.

If implementation details are important for maintainers, place them in comments rather than
docstrings.

---

# Maintain Accuracy

Whenever changing behavior:

- update the corresponding docstring
- update examples
- update documented exceptions
- update return value descriptions if necessary

Outdated documentation is often more harmful than missing documentation.

---

# Cross References

When appropriate, reference related functionality using Julia's documentation conventions.

Examples include:

- related operators
- complementary types
- alternative constructors

Use cross references to help users discover the API rather than repeating documentation.

---

# Writing Style

Write in clear, concise English.

Prefer:

- active voice
- complete sentences
- consistent terminology
- precise mathematical language

Avoid:

- unnecessary verbosity
- implementation-specific language
- ambiguous wording
- unexplained abbreviations

Use the same terminology consistently throughout the codebase.

---

# Documentation Checklist

Before opening a PR, verify that:

- every new public API has a docstring
- docstrings follow the existing project style
- `# Arguments`, `# Returns`, `# Exceptions`, and `# Examples` sections are included where
  appropriate
- examples are valid `jldoctest`s when provided
- mathematical terminology is accurate
- documentation matches the implementation
- related documentation is updated when behavior changes

---

# Things to Avoid

Do not:

- document the implementation instead of the interface
- repeat obvious type information without explanation
- leave stale examples after refactoring
- duplicate identical documentation across multiple methods
- omit documentation for new public APIs
- include speculative or future behavior

Prefer documentation that explains the purpose and semantics of an API over documentation
that merely describes its syntax.
