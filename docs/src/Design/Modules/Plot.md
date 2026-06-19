```@meta
CurrentModule = Mantis.Plot
```
# Plot

The `Plot` module is how you get results out of `Mantis` for visualisation. There are two
routes: exporting to VTK files for an external viewer such as
[ParaView](https://www.paraview.org/), and, through an optional extension, drawing
one-dimensional solutions directly with [Makie](https://docs.makie.org/).

!!! note "It writes files, it does not render"
    The core `Plot` module does not draw anything itself; it writes VTK files. In-process
    plotting is provided only by the Makie extension described below.

## Exporting to VTK

The main entry points produce `.vtu`/`.vtk` files. Because high-degree and curved elements
cannot be represented exactly by straight-sided cells, the exporters sample each element at a
chosen polynomial `degree` and optionally subdivide it (`n_subcells`), exporting high-order
Lagrange cells that ParaView can refine further:

- [`plot`](@ref) exports a geometry, or a form/field, to VTK.
- [`export_form_fields_to_vtk`](@ref) exports one or more form fields (e.g. a computed solution
  and its exact counterpart) into VTK files.
- [`export_geometry_to_vtk`](@ref) exports just a geometry, which is useful for checking a mesh
  or a mapping.

```julia
# ϕ⁰ is a computed solution, ϕ_exact an AnalyticalFormField on the same geometry
Mantis.Plot.export_form_fields_to_vtk((ϕ⁰, ϕ_exact), "my_solution")
```

The [Hodge Laplacian](@ref) and [Biharmonic](@ref) examples both end with a VTK export step.

## Drawing with Makie

For quick, in-notebook visualisation of **one-dimensional** solutions, `Mantis` provides
`plot_solution`, which evaluates a tuple of form fields over the mesh and returns a Makie
figure. It lives in a package extension and is therefore only available once a Makie backend
has been loaded:

```julia
using Mantis
using GLMakie   # or CairoMakie; this activates the MakieExt extension

fig = Mantis.Plot.plot_solution((ϕ⁰, ϕ_exact); title = "Solution")
```

Because it depends on Makie being present, `plot_solution` is intentionally *not* part of the
hard dependencies of `Mantis`; without a Makie backend loaded the method does not exist. The
[Heat Equation](@ref) and [Biharmonic](@ref) examples use it to plot basis functions and
solutions. For higher-dimensional fields, prefer the VTK route above.

## All docstrings from Mantis.Plot
```@autodocs
Modules = [Main.Mantis.Plot]
```
