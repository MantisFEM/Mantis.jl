---
name: Mantis Example Documentation
description: Write and maintain example documentation pages that demonstrate how to use Mantis.jl through complete, executable examples, following the style used throughout `docs/src/Examples/`.
---

# Mantis.jl Example Documentation

This skill describes how to write or update example pages under:

```
docs/src/Examples/
```

Examples are intended to teach users how to solve problems using Mantis, not to document
individual APIs.

Each example should demonstrate a complete workflow that users can adapt to their own
problems.

---

# Purpose

An example should answer questions such as:

- How do I use this functionality?
- What is the typical workflow?
- Which abstractions work together?
- What result should I expect?

Examples should be practical, self-contained, and easy to adapt.

---

# Source Files

Documentation pages are generated from

```
examples/src/*.jl
```

using Literate.jl.

**Always edit the `.jl` source file**, not the generated Markdown page.

Ensure that both the code and the accompanying comments are suitable for Literate.jl.

---

# Intended Audience

Assume readers:

- are familiar with Julia,
- have basic knowledge of finite element methods,
- are learning how to use Mantis.

Explain Mantis-specific concepts, but avoid explaining general Julia syntax.

---

# Structure

A typical example should contain:

1. Introduction
2. Problem description
3. Geometry construction
4. Function space definition
5. Weak formulation or computation
6. Results
7. Discussion

Not every example requires every section, but the progression should feel natural.

Whenever possible, build on previous examples to avoid repeating code.

---

# Introduce the Problem

Begin by describing:

- the mathematical problem,
- the objective of the example,
- the important Mantis features being demonstrated.

Keep the introduction concise.

---

# Build the Example Incrementally

Construct the example in logical steps.

For example:

- define the geometry,
- define function spaces,
- construct forms,
- assemble the problem,
- solve,
- analyze or visualize the results.

Avoid presenting a large block of code without explanation.

---

# Explain Design Decisions

Describe *why* each step is performed.

For example:

- why a particular geometry is chosen,
- why a certain function space is appropriate,
- why a specific operator is used.

Help readers understand the reasoning behind the workflow.

---

# Focus on Public APIs

Examples should use the public Mantis interface.

Avoid relying on:

- internal helper functions,
- undocumented interfaces,
- implementation details.

Examples should remain valid as the implementation evolves.

---

# Keep Examples Realistic

Use examples that resemble actual workflows.

Prefer:

- standard geometries,
- commonly used function spaces,
- realistic boundary conditions,
- representative numerical problems.

Avoid contrived examples that only demonstrate syntax.

---

# Keep Examples Concise

Include enough code to illustrate the workflow, but avoid unnecessary complexity.

If an example becomes too long, consider splitting it into multiple examples.

Each example should focus on one primary concept.

---

# Explain the Mathematics

Briefly describe the mathematical formulation when it improves understanding.

For example:

- the governing equations,
- weak formulations,
- differential operators,
- finite element spaces.

Keep mathematical discussions concise and directly relevant to the example.

---

# Results

Conclude by explaining:

- what was computed,
- what users should observe,
- how to interpret the output.

When appropriate, discuss expected numerical behavior or convergence properties.

---

# Visualizations

Include plots or VTK output only when they help explain the result.

Visualizations should support the narrative, not replace it.

Ensure generated figures are reproducible.

---

# Cross References

Link to related documentation where appropriate.

Examples include:

- module documentation,
- tutorials,
- API documentation,
- related examples.

Avoid duplicating explanations that already exist elsewhere.

---

# Maintain Examples

Whenever the implementation changes:

- update the example,
- update explanatory text,
- update expected output,
- remove obsolete code.

Examples should always execute successfully.

---

# Writing Style

Write in clear, concise English.

Use comments to explain concepts rather than individual Julia statements.

Prefer explaining the workflow over explaining syntax.

---

# Example Checklist

Before opening a PR, verify that:

- the example is implemented in `examples/src/`
- the example follows the style of existing examples
- the workflow is easy to follow
- only public APIs are used
- explanations focus on concepts rather than syntax
- mathematical terminology is accurate
- the example executes successfully
- generated documentation renders correctly with Literate.jl
- visualizations, if included, are reproducible

---

# Things to Avoid

Do not:

- edit the generated Markdown files
- rely on undocumented or internal APIs
- include large unexplained code blocks
- explain basic Julia syntax
- mix multiple unrelated concepts into a single example
- leave outdated output or figures after implementation changes

Prefer examples that teach users how to solve realistic problems using Mantis rather than
examples that merely demonstrate individual functions.
