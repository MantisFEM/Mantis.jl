---
name: Mantis Module Documentation
description: Write and maintain module documentation pages that explain the mathematical concepts, architecture, and public interfaces of a Mantis.jl module, following the style used in `docs/src/Design/Modules/`.
---

# Mantis.jl Module Documentation

This skill describes how to write or update module documentation pages under

```
docs/src/Design/Modules/
```

These pages are intended to explain the design of a module to users and contributors, rather
than documenting individual APIs.

Use existing pages, such as `docs/src/Design/Modules/Forms.md`, as the primary style
reference.

---

# Purpose

A module page should answer questions such as:

- What problem does this module solve?
- What mathematical concepts does it implement?
- What role does it play within Mantis?
- How does it interact with other modules?
- What are its primary public abstractions?
- How should users think about using it?

The emphasis is on **concepts and design**, not implementation details.

---

# Intended Audience

Assume readers:

- are familiar with Julia,
- have some background in finite element methods,
- may be unfamiliar with the internal architecture of Mantis.

Explain mathematical concepts when they are central to understanding the module, but avoid
turning the page into a textbook.

---

# Structure

Follow the structure used by existing module pages.

A typical page contains:

1. Overview
2. Core concepts
3. Main abstractions
4. Relationships to other modules
5. Examples
6. References (when appropriate)

Not every page requires every section, but the overall organization should remain consistent
across modules.

---

# Overview

Begin with a concise overview describing:

- the module's responsibility,
- where it fits within the overall architecture,
- what kinds of functionality it provides.

Avoid discussing implementation files or internal organization.

Focus on the conceptual role of the module.

---

# Explain the Mathematics

Many Mantis modules represent mathematical abstractions.

Describe:

- the mathematical objects represented,
- the notation used throughout the library,
- important invariants,
- the relationship between mathematical concepts and their implementation.

For example:

- differential forms
- pullbacks
- Hodge operators
- finite element spaces
- hierarchical refinement

Use precise mathematical terminology.

---

# Explain the Design

Describe the design decisions behind the module.

For example:

- why symbolic expressions are used,
- why evaluation is lazy,
- why canonical coordinates are preferred,
- how abstraction boundaries are maintained.

Help readers understand *why* the module is designed the way it is.

---

# Introduce Public Abstractions

Present the important public types and interfaces.

Explain:

- what they represent,
- when they should be used,
- how they relate to one another.

Do not duplicate API docstrings. Instead, extensively embed the docstrings into the
documentation page, creating a cohesive narrative.

---

# Describe Module Relationships

Explain how the module interacts with the rest of Mantis.

Examples include:

- Geometry provides mappings used by Forms.
- FunctionSpaces provide basis functions consumed by Forms.
- Assemblers evaluate symbolic expressions built by Forms.

Help readers understand the flow of information through the library.

---

# Use Examples

Prefer small conceptual examples over exhaustive tutorials.

Examples should illustrate:

- typical workflows,
- important abstractions,
- common usage patterns.

Avoid large blocks of implementation code.

---

# Maintain Conceptual Level

Keep the documentation focused on design.

Prefer explaining:

- why something exists,
- what abstraction it provides,
- how it fits into the library.

Avoid describing:

- internal helper functions,
- implementation details,
- private data structures,
- optimization techniques unless they motivate the design.

---

# Cross References

Link to related documentation where appropriate.

Examples include:

- other module pages,
- tutorials,
- examples,
- API documentation.

Use cross references to help readers navigate the documentation rather than duplicating
content.

---

# Mathematical Notation

When introducing notation:

- define symbols before using them,
- use consistent notation throughout the page,
- match notation used elsewhere in the documentation.

Prefer mathematical clarity over excessive formalism.

---

# Diagrams

When appropriate, include simple diagrams illustrating:

- module relationships,
- data flow,
- mathematical structures,
- abstraction layers.

Diagrams should clarify the design rather than decorate the page.

---

# Keep Documentation Current

Whenever the module changes:

- update the conceptual description,
- update examples,
- update descriptions of public abstractions,
- update relationships to other modules.

Documentation should evolve together with the implementation.

---

# Documentation Checklist

Before opening a PR, verify that:

- the page follows the style of existing module documentation (especially
  `docs/src/Design/Modules/Forms.md`)
- the module's purpose is clearly explained
- the mathematical concepts are described accurately
- the main abstractions are introduced
- relationships to other modules are explained
- examples are concise and illustrative
- implementation details are minimized
- terminology is consistent with the rest of the documentation
- cross references are updated where appropriate

---

# Things to Avoid

Do not:

- duplicate API docstrings
- document individual methods in detail
- explain the implementation file layout
- include unnecessary implementation details
- introduce notation without defining it
- turn the page into a complete mathematical reference

Prefer documentation that explains the *design philosophy* and *conceptual model* of the
module over documentation that mirrors the source code.
