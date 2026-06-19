```@meta
CurrentModule = Mantis.Mesh
```
# Mesh

The `Mesh` module holds the *combinatorial* description of a discretisation: how the domain is
partitioned into elements, and how those elements, faces, edges and vertices connect to one
another. The [Geometry](@ref DocGeometryModule) module says where each element sits in physical
space; the `Mesh` module says how the elements fit together.

!!! note "Internals may change"
    As stated in the module docstring, several `Mesh` structs are still simple scaffolding to
    support experimentation in the rest of the code, and their internals (even fields not
    prefixed with an underscore) may change without warning. Prefer the getter functions over
    reaching into fields directly.

## Patches

A *patch* records the breakpoints (element boundaries) of a structured block of the mesh:

- [`Patch1D`](@ref) stores a strictly-increasing vector of breakpoints for a 1D block.
- [`Patch`](@ref){`n`} is the `n`-dimensional, tensor-product generalisation.

Helpers such as `get_breakpoints`, `get_element_vertices` and `get_element_measure` read
geometric information about individual elements out of a patch.

## Topology

[`MeshTopology`](@ref) captures the incidence relations of a mesh: which faces bound which
cells, which edges bound which faces, and so on. These relations underpin operations that need
neighbour information, for example the `compute_face_neighbours`, `compute_edge_neighbours` and
`compute_vertex_neighbours` routines. They are mostly used internally by the higher-level
spaces and geometries, but are exposed here for advanced use.

## All docstrings from Mantis.Mesh
```@autodocs
Modules = [Main.Mantis.Mesh]
```
