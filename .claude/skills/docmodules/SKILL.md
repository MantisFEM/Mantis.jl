---
name: Mantis Module Documentation
description: Write or update a module design page under `docs/src/Design/Modules/` for a Mantis.jl module — explaining its purpose, mathematical concepts, public abstractions, and relationships to other modules. Use this whenever the user asks to document a module, write or improve a module page, fill in a stub or `@autodocs` placeholder, describe what a module does or how it fits into Mantis, or refresh module docs after a refactor. Reach for it even when they only say something like "document Quadrature" or "the Geometry docs are out of date" without naming the docs directory.
---

# Mantis.jl Module Documentation

Module pages under `docs/src/Design/Modules/` explain **what a module represents and why its
abstractions exist**. They sit between the theory pages and the API reference: a reader who
knows FEM but not Mantis should come away with the module's mental model.

This is not an API dump. Docstrings are *woven into* a narrative, not listed. If the page
reads like a table of contents for the source tree, it has failed.

`docs/src/Design/Modules/Forms.md` is the canonical style reference; `Points.md` is the
cleanest short example of the same style. Read at least one before writing.

## Before writing

Documentation that drifts from the code is worse than none, so start from the source:

1. **Read the module.** The types, their parameters, and the existing docstrings. Type
   parameters usually *are* the mental model — `AbstractForm{manifold_dim, form_rank,
   expression_rank}` tells you what a form is in Mantis.
2. **Inventory what must be covered.** `makedocs` is configured with `modules=[...]`, which
   makes Documenter **fail the build if any docstring from the module is missing from the
   docs**. List the module's public names and plan for each to land in some `@docs` block.
   This is why unwritten pages use `@autodocs` — it satisfies the check trivially. Replacing
   such a stub means accounting for everything it was silently covering.
3. **Check the existing page.** Several are stubs (`Quadrature.md`, `Mesh.md`,
   `FunctionSpaces.md`); a few are substantial. Keep whatever is already accurate.
4. **Find the neighbours.** Which modules does it depend on, and which depend on it? Skim
   their pages so cross-references and terminology line up.

Infer the module and depth from the request; ask only when genuinely ambiguous.

## Page anatomy

A page sets the module for docstring lookup, then a title carrying the page anchor
`Doc<Module>Module`, then the module's own docstring:

````markdown
```@meta
CurrentModule = Mantis.Points
```
# [Points](@id DocPointsModule)

```@docs
Points
```

## Overview
...
````

A page may switch `CurrentModule` partway through — `TensorProducts.md` starts at `Mantis`
to document the module itself, then narrows — so do that when you need names from two scopes.

Section headings carry anchors named `<Module><Topic>` in CamelCase: `PointsAbstract`,
`PointsConcrete`, `FormsCreation`, `TensorProductsRequirement`. Add an anchor to any section
another page will want to link to; sections nobody links to don't need one.

Typical section flow, adapted to the module rather than applied mechanically:

1. **Overview** — the problem the module solves and its role in Mantis
2. **Core concepts** — the mathematics, only as deep as the design requires
3. **Main abstractions** — the public types, in the order a reader meets them
4. **Relationships to other modules** — what flows in and out
5. **Examples** — short and conceptual
6. **References** — when the module implements published work

`Points.md` follows almost exactly this. `Forms.md` instead organises around *creating* forms
and then *operating on* them, because that is how users actually think about forms. Follow
the module's own logic rather than forcing the list.

## Writing the narrative

**Lead with the problem, not the type.** `Points.md` opens by explaining that evaluation
happens in the canonical domain ``[0, 1]^n`` and that the module exists to keep the meaning
of "a point" consistent across the library. Only then does `AbstractPoints` appear. A reader
who understands the problem can half-predict the abstraction, which is what makes it stick.

**Weave docstrings into prose.** Introduce what a type is for, include it, then connect it to
what comes next:

````markdown
All point sets in Mantis are subtypes of `AbstractPoints`:

```@docs
AbstractPoints
```

The module requires every concrete subtype to implement `get_num_points`:

```@docs
get_num_points
```

From this, other methods can be automatically derived, without needing manual
implementation:

```@docs
get_manifold_dim
get_input_points
```
````

Group related names into one block when a single sentence introduces them all; give a name
its own block when it deserves its own introduction. Never restate what the docstring already
says — if you catch yourself paraphrasing one you just included, cut the prose (or fix the
docstring).

Each docstring may appear in **exactly one** `@docs` block across the whole documentation, or
the build errors on a duplicate. When a name belongs to another module's page, link instead:
``[`FunctionSpaces.AbstractFESpace`](@ref)``.

**Explain design choices where they illuminate the abstraction.** Why forms evaluate in
canonical coordinates and pull back; why `PointSet` stores coordinates dimension-wise, so
tensor-product routines can take a whole dimension in one access without allocating; why
`TensorProducts` centralises index bookkeeping instead of letting each module reinvent it.
These are the sentences readers remember. A choice with no user-visible consequence is
implementation detail — leave it out.

**Mark internals explicitly** when a reader genuinely benefits from knowing them, using the
established admonition:

````markdown
### [Internals: How a `FormSpace` is evaluated](@id FormsInternalEvaluateFormSpace)
!!! note "Internal behaviour"
    We explain how a `FormSpace` is evaluated. However, this is considered an
    implementational detail.
````

Use it sparingly, for behaviour that shapes how users reason about results — not as licence
for a tour of private helpers.

## Mathematics and notation

Define notation before using it, and match what the theory pages and neighbouring module
pages already use: ``n`` for manifold dimension, ``m`` for embedding dimension, ``\Omega^0
:= [0, 1]^n`` for the canonical domain, ``\Lambda^k_h`` for discrete form spaces, ``\Phi``
for geometry mappings.

Inline math uses double backticks (``` ``[0, 1]^n`` ```); displayed math uses ` ```math `
blocks. Include only the mathematics the design rests on. `Geometry.md` develops the
tensor-product geometry and its Jacobian in full because the module's entire structure
follows from it — that depth is earned, not a default.

## Examples

Prefer one short `@repl` session showing the abstraction doing its job. Name the session so
several blocks can share state:

````markdown
```@repl CreatingFormSpaces
using Mantis
B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (4, 4), (3, 3), (2, 2))
```
we can use this function space to create two different spaces: one for a ``0``-form
``\Lambda^0_h`` and one for a ``2``-form ``\Lambda^2_h``.
```@repl CreatingFormSpaces
Λ⁰ₕ = Forms.FormSpace(0, B, "0-form")
Λ²ₕ = Forms.FormSpace(2, B, "2-form")
```
````

`@repl` blocks **execute during the build**, so every line must run against the current API.
Verify them instead of writing plausible-looking code. Full workflows belong in
`docs/src/Examples/` — link there rather than reproducing them.

## Cross-references and citations

- Module page: `[Geometry](@ref DocGeometryModule)`, or `[Quadrature](@ref)` when the page
  title alone resolves
- Section: `[Operations on Forms](@ref FormsOperations)`
- API name: ``[`FunctionSpaces.AbstractFESpace`](@ref)``
- Theory page: `[differential form theory page](@ref TheoryForms)`
- Literature: `[Arnold2010](@cite)` or `[Bezanson2017, Bezanson2018](@cite)`, with keys in
  `docs/src/refs.bib`

Link rather than duplicate: when your page needs a concept another page owns, one sentence
plus a reference beats a paragraph of restatement.

A "Relationships to Other Modules" section works well as a short bulleted list, each entry
naming what is given or taken:

````markdown
- **[Geometry](@ref DocGeometryModule)**: evaluates geometries by mapping points in the
  canonical domain to a set of points on a given element.
- [Quadrature](@ref): quadrature rules store their nodes as `AbstractPoints`.
````

Add a diagram only when the architecture or data flow is genuinely hard to follow in prose.

## Registering and verifying the page

A new page is invisible until listed in `docs/make.jl`, in the `Modules` entry of `Design`
(alphabetical) and — if the module isn't there yet — in `makedocs(modules=[...])`. Both are
easy to forget and both fail confusingly.

Then build, because every failure mode here is a build-time one:

```bash
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```

Watch for docstrings from the module that no block includes, the same docstring included
twice, unresolved `@ref` targets, and `@repl` blocks that error. If the build breaks for
reasons unrelated to your page, say so plainly rather than reporting the page as verified.

The block and anchor tables at the end of this page have the exact syntax when you need it.

## What to avoid

- Organising sections around source files, or mentioning file layout at all
- Listing docstrings with no connecting prose, or paraphrasing docstrings you just included
- Documenting private helpers, internal data structures, or method-by-method behaviour
- Mathematical development beyond what the design needs
- Notation introduced without definition, or diverging from the rest of the docs
- Restating what a tutorial, example, or theory page already owns

## Final check

- Follows `Forms.md` / `Points.md`: `@meta` header, `Doc<Module>Module` anchor, module
  docstring, narrative with woven `@docs` blocks
- Purpose and mathematical concepts are clear to someone new to Mantis
- Every public name of the module appears in exactly one `@docs` block somewhere in the docs
- Relationships to other modules are explained and cross-referenced
- Examples are short, conceptual, and actually run
- Terminology and notation match the surrounding documentation
- Page is registered in `docs/make.jl` and the docs build

## Documenter reference

### Blocks

| Block | Purpose | Notes |
|---|---|---|
| ` ```@meta ` | Set `CurrentModule` for unqualified docstring lookup | First block on the page; may be repeated to switch scope mid-page |
| ` ```@docs ` | Include named docstrings | One name per line. Each docstring may appear only once in the whole doc set |
| ` ```@autodocs ` | Include every docstring of a module | What unwritten stub pages use; replace with narrative `@docs` blocks |
| ` ```@repl <name> ` | Executed REPL session, output shown | Runs at build time. Named sessions share state across blocks |
| ` ```math ` | Displayed equation | Inline math uses double backticks instead |
| `!!! note "..."` | Admonition | The established one is `!!! note "Internal behaviour"` |

Disambiguating an overloaded method inside a `@docs` block needs the full signature:

```
evaluate(::AbstractForm{manifold_dim}, ::Int, ::Points.AbstractPoints{manifold_dim}) where {manifold_dim}
```

Callable structs (functors) can trip Documenter's docstring check — see the comment in
`docs/make.jl`; the workaround is to attach the docstring to the type definition.

### Anchors

| Kind | Pattern | Examples |
|---|---|---|
| Module page title | `Doc<Module>Module` | `DocPointsModule`, `DocGeometryModule`, `DocTensorProductsModule` |
| Section within a page | `<Module><Topic>` | `PointsAbstract`, `PointsConcrete`, `FormsCreation`, `TensorProductsRequirement` |
| Theory pages | descriptive | `TheoryForms`, `FEEC`, `ExtCoeffs` |

`Assemblers.md` declares `DocAssemblyModule`, not `DocAssemblersModule`. Match whatever a
page already declares rather than "correcting" it — other pages link to it.

### Page skeleton

````markdown
```@meta
CurrentModule = Mantis.<Module>
```
# [<Module>](@id Doc<Module>Module)

```@docs
<Module>
```

## Overview

<The problem the module solves and its role in Mantis. Define any notation the page relies
on, such as the canonical domain.>

## [<Core concept>](@id <Module>Concepts)

<The mathematics, to the depth the design requires.>

## [<Main abstraction>](@id <Module>Abstract)

<Introduce the type, include it, then relate it to what follows.>

```@docs
Abstract<Thing>
```

## [Relationships to Other Modules](@id <Module>Relationships)

- **[<Other module>](@ref Doc<Other>Module)**: <what flows between them>

## References

<Only when the module implements published work: [Key](@cite).>
````
