# Mantis.jl

Julia package for high-order, structure-preserving finite element methods, built on the
Finite Element Exterior Calculus (FEEC) framework. Supports piecewise-polynomial,
non-polynomial, and adaptively-refinable (hierarchical B-spline) finite element spaces,
among others.

## Build & test Supports Julia 1.10 (LTS) through pre-release (CI runs lts, 1, pre).

```
bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'          # resolve deps
julia --project=. -e 'using Pkg; Pkg.test()'                  # full suite
julia --project=. -e 'include("test/Geometry/runtests.jl")'   # one module
julia --project=. -e 'include("test/Assemblers/GlobalAssemblersTests.jl")'  # one file
```

Test files use plain import Mantis / top-level @tests (no @testset inside each file);
@testset wrapping happens in each module's test/<Module>/runtests.jl, which is in turn
included from test/runtests.jl. Mirror src/<Module> structure when adding tests, and
register new files in the module's runtests.jl.
JET type-inference tests (test/Inference) only run on Julia >= 1.12, non-prerelease, via
their own nested project — skipped otherwise.

Docs: julia --project=docs docs/make.jl.

Example pages under docs/src/Examples are generated from examples/src/\*.jl via Literate.jl
— edit the .jl source, not the generated .md.

`jlfmt` is used to enforce consistent formatting.

## Architecture src/Mantis.jl includes each submodule in dependency

order and re-exports selected symbols. Modules depend roughly on those before them:
GeneralHelpers -> Mesh -> Points -> Hierarchy -> Geometry -> FunctionSpaces -> Quadrature ->
Forms -> Analysis -> Assemblers -> Plot

Each module's <ModuleName>.jl defines the module, abstract types, exports, and top-level
fallback methods (throw(MethodError(...)) documenting the interface concrete subtypes must
implement), then includes the rest of the module at the bottom — read it first when
exploring a module.

- **Mesh**: topology-only connectivity, no geometry.
- **Points**: point sets used for evaluation (e.g. CartesianPoints).
- **Hierarchy**: bookkeeping for hierarchically-refined (adaptive) structures.
- **Geometry**: AbstractGeometry{manifold_dim, image_dim, num_patches} maps the canonical
  domain [0,1]^manifold_dim to physical space. Evaluation happens in the canonical domain,
  not physical space — this distinction carries through Forms and Assemblers too.
- **FunctionSpaces**: AbstractFESpace and concrete FE spaces (e.g. B-splines).
- **Quadrature**: element-level and global quadrature rules. -
  **Forms**: the FEEC layer. AbstractForm{manifold_dim, form_rank, expression_rank} is
  central — expression_rank 0 = field (no basis), 1 = space (has a basis), 2 = two-basis
  expression (e.g. from wedge products). Operators (d, ★, ♯, ∧, ∫, dstar/δ) build expression
  trees rather than eagerly computing values; evaluate then walks the tree.
- **Analysis**: error computation (e.g. L2 error).
- **Assemblers**: turns Forms expressions into global sparse matrices/vectors
  (GlobalAssemblers.jl + per-problem WeakFormulations).
- **Time Integrators**: Use for solving time-dependent problems. Uses a GLM framework.
- **Plot**: writes VTK output; Makie plotting lives in the weak-dependency extension
  ext/MakieExt.jl. exports/Exports.jl and exports/Forms.jl are auxiliary re-exports of the
  curated public API (marked "auto-generated") — keep in sync manually if you add/rename
  public Forms exports.

## Conventions

- **Style**: [Blue Style](https://github.com/JuliaDiff/BlueStyle) with the modifications in
  .JuliaFormatter.toml (no import->using conversion, short function defs stay short,
  markdown/docstring formatting disabled, docs/ excluded).
- **Docstrings**: public functions/types get # Arguments, # Returns (when arguments are not
  clear), and where relevant # Exceptions sections, matching existing src/ style. Add #
  Examples with a jldoctest block when it aids understanding.
- **Commit messages** are enforced by .github/scripts/commit-msg.sh / CommitMessage.yml CI
  (Conventional Commits): <type>: <text> (<type>!: <text> for breaking changes, revert:
  <type>: <text> for reverts). Allowed types: bench, chore, ci, doc, feat, fix, perf,
  refactor, style, test, revert, merge. Title <= 50 chars; first word of <text> must be
  lower-case, imperative/future tense (not -ed/-s/-ing), e.g. feat: add new cool geometry.
  Keep commits single-purpose — see docs/src/Support/Contributing.md (e.g. perf: commits
  should quantify improvement with something like @allocations).
- **Public-but-undocumented exports** use Julia's public keyword (guarded by VERSION >=
  v"1.11.0-DEV.469" for Julia 1.10 compatibility) instead of export, e.g. abstract types in
  Forms.jl.
- **Weak dependencies**: go through a package extension in ext/, declared via
  [weakdeps]/[extensions] in Project.toml, not a hard dependency.
