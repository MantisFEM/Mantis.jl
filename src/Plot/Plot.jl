"""
This (sub-)module provides a collection of plotting routines for Mantis evaluatable structs.

The exported names are:
"""

module Plot
import WriteVTK 

import .. Geometry

include("./PlotKernel.jl")

@doc raw"""
    plot(geometry::G; kwargs...) where {G <: AbstractGeometry}

Plot a geometry `geometry` by generating a vtk file containing the definition of the (high polynomial) 
degree elements.

# Arguments
- `geometry::AbstractGeometry`: the geometry to plot.

# Keyword arguments
- `vtk_filename::String`: The name of the vtk file where to save the plot (`.vtk` extension is automatically added). 
   <default>: `"default"`.
- `n_subcells::Int64`: The number of subdivisions (per dimension) to generate in each element, i.e., each 
  element is exported as `n_subcells^d` cells, where `d` is the geometric dimension. This allows for a more
  detailed plotting when using basis functions with polynomial degree higher than one. <default>: `1`. 
- `degree::Int64`: The polynomial degree of the exported vtk cells. Each element is sampled at `(degree + 1)^d`, 
  where `d` is the geometric dimension, points and these are exported as a high degree `VTK_LAGRANGE_QUADRILATERAL`.
  Highly refined visualization is then available in Paraview, for example, by adding an `ExtractSurface` 
  filter and then increasing the `Nonlinear Subdivision Level`. Each subdivision level in Paraview doubles the number of 
  visualization cells in each dimension.

# Returns 
Nothing
"""
function plot(geometry::Geometry.AbstractGeometry{n, m}; kwargs...) where {n, m}
    _plot(geometry; kwargs...)
end


@doc raw"""
plot(field::F; kwargs...) where {F <: AbstractFunction}

Plot a field `field` by generating a vtk file containing the definition of the (high polynomial) 
degree function.

Note that both the underlying geometry of the field and the field itself are plotting in an isoparametric way.

# Arguments
- `geometry::AbstractGeometry`: the geometry to plot.

# Keyword arguments
- `vtk_filename::String`: The name of the vtk file where to save the plot (`.vtk` extension is automatically added). 
   <default>: `"default"`.
- `n_subcells::Int64`: The number of subdivisions (per dimension) to generate in each element, i.e., each 
  element is exported as `n_subcells^d` cells, where `d` is the geometric dimension. This allows for a more
  detailed plotting when using basis functions with polynomial degree higher than one. <default>: `1`. 
- `degree::Int64`: The polynomial degree of the exported vtk cells. Each element is sampled at `(degree + 1)^d`, 
  where `d` is the geometric dimension, points and these are exported as a high degree `VTK_LAGRANGE_QUADRILATERAL`.
  Highly refined visualization is then available in Paraview, for example, by adding an `ExtractSurface` 
  filter and then increasing the `Nonlinear Subdivision Level`. Each subdivision level in Paraview doubles the number of 
  visualization cells in each dimension.

# Returns 
Nothing
"""
# function plot(field::F; kwargs...) where {F <: Function}
#     _plot(field.geometry, field; kwargs...)
# end

end