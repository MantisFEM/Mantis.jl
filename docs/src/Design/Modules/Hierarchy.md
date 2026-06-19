```@meta
CurrentModule = Mantis.Hierarchy
```
# Hierarchy

The `Hierarchy` module provides the bookkeeping that makes *adaptive refinement* possible. A
hierarchically-refined discretisation is organised into a sequence of nested levels (level 1
the coarsest), and at any point only a subset of the objects on each level is *active*. The
`Hierarchy` module tracks which objects (elements or basis functions) are active on each
level, and provides a stable mapping between two ways of identifying them:

- a flat **hierarchical id** that runs across all levels, and
- a **(level, level-local id)** pair.

This information is bundled in [`ActiveInfo`](@ref), the central type of the module. Given an
`ActiveInfo`, you can ask for the active ids on a level, the number of active objects, the
level a given hierarchical id belongs to (`get_level`), and convert a hierarchical id to its
level-local id (`convert_to_level_id`).

`ActiveInfo` is the connective tissue between the adaptively-refinable
[`HierarchicalFiniteElementSpace`](@ref Mantis.FunctionSpaces.HierarchicalFiniteElementSpace)
in the [FunctionSpaces](@ref) module and the
[`HierarchicalGeometry`](@ref Mantis.Geometry.HierarchicalGeometry) in the
[Geometry](@ref DocGeometryModule) module: both describe their active sets in terms of it. You
rarely build an
`ActiveInfo` by hand; the refinement routines produce and update it. It is the object passed
around to describe the current state of refinement. The
[Adaptive refinement](@ref) example shows it being threaded through an adaptive loop.

## All docstrings from Mantis.Hierarchy
```@autodocs
Modules = [Main.Mantis.Hierarchy]
```
