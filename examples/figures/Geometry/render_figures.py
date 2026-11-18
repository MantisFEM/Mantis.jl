# Renders the figures of the Geometry example with ParaView.
#
# Usage (normally called by make_figures.jl):
#   pvbatch render_figures.py <vtk_dir> <png_dir>
#
# <vtk_dir> must contain the VTK files written by the example.

import os
import sys

from paraview.simple import (
    a3DText,
    AssignViewToLayout,
    Calculator,
    ColorBy,
    ExtractSurface,
    CreateLayout,
    CreateRenderView,
    Delete,
    GenerateIds,
    GetColorTransferFunction,
    GetScalarBar,
    Show,
    Text,
    Transform,
    Tube,
    XMLPolyDataReader,
    XMLUnstructuredGridReader,
    SaveScreenshot,
)
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData
from vtkmodules.vtkIOXML import vtkXMLPolyDataWriter, vtkXMLUnstructuredGridReader

vtk_dir, png_dir = sys.argv[1], sys.argv[2]
os.makedirs(png_dir, exist_ok=True)

WHITE = [1.0, 1.0, 1.0]
BLACK = [0.0, 0.0, 0.0]
EDGE_COLOR = [0.15, 0.15, 0.15]
FONT_SIZE = 22


def vtk_file(name):
    return os.path.join(vtk_dir, name + ".vtu")


def new_view(size, parallel=False):
    view = CreateRenderView()
    view.ViewSize = size
    view.OrientationAxesVisibility = 0
    try:
        view.UseColorPaletteForBackground = 0
    except AttributeError:
        pass
    view.Background = WHITE
    view.CameraParallelProjection = 1 if parallel else 0
    return view


def with_element_index(source):
    # With n_subcells = 1 (the default), the VTK cells are the elements, in order.
    ids = GenerateIds(Input=source)
    return Calculator(
        Input=ids,
        AttributeType="Cell Data",
        ResultArrayName="element",
        Function="CellIds + 1",
    )


def show_elements(source, view, num_elements, scalar_bar=False):
    display = Show(with_element_index(source), view)
    display.NonlinearSubdivisionLevel = 3
    display.Ambient = 0.25
    display.Diffuse = 0.85
    ColorBy(display, ("CELLS", "element"))
    lut = GetColorTransferFunction("element")
    lut.ApplyPreset("Cool to Warm", True)
    lut.RescaleTransferFunction(1.0, float(num_elements))
    lut.AutomaticRescaleRangeMode = "Never"
    display.SetScalarBarVisibility(view, scalar_bar)
    if scalar_bar:
        bar = GetScalarBar(lut, view)
        bar.Title = "element index"
        bar.ComponentTitle = ""
        bar.TitleColor = BLACK
        bar.LabelColor = BLACK
        bar.TitleFontSize = FONT_SIZE
        bar.LabelFontSize = FONT_SIZE - 4
        bar.RangeLabelFormat = "%-#.0f"
        bar.WindowLocation = "Any Location"
        bar.Orientation = "Vertical"
        bar.Position = [0.88, 0.15]
        bar.ScalarBarLength = 0.7
    return display


def show_wireframe(name, view, width=2.0):
    display = Show(XMLUnstructuredGridReader(FileName=[vtk_file(name + "_wireframe")]), view)
    display.ColorArrayName = ["POINTS", ""]
    display.NonlinearSubdivisionLevel = 3
    display.AmbientColor = EDGE_COLOR
    display.DiffuseColor = EDGE_COLOR
    display.LineWidth = width
    return display


def show_title(text, view):
    title = Text(Text=text)
    display = Show(title, view)
    display.WindowLocation = "Upper Center"
    display.FontSize = FONT_SIZE
    display.Color = BLACK
    return display


def element_endpoints(name, offset=(0.0, 0.0, 0.0)):
    # The first two points of each VTK Lagrange curve are its end points, i.e. the element
    # vertices. They are written to a .vtp file so that ParaView can draw them as spheres.
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtk_file(name))
    reader.Update()
    grid = reader.GetOutput()
    points = vtkPoints()
    vertices = vtkCellArray()
    for cell_id in range(grid.GetNumberOfCells()):
        cell = grid.GetCell(cell_id)
        for local in (0, 1):
            x = grid.GetPoint(cell.GetPointId(local))
            point_id = points.InsertNextPoint([x[i] + offset[i] for i in range(3)])
            vertices.InsertNextCell(1)
            vertices.InsertCellPoint(point_id)
    polydata = vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    filename = os.path.join(vtk_dir, name + "_endpoints.vtp")
    writer = vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()
    return XMLPolyDataReader(FileName=[filename])


def show_curve(name, view, num_elements, offset=(0.0, 0.0, 0.0), radius=0.03):
    source = XMLUnstructuredGridReader(FileName=[vtk_file(name)])
    if offset != (0.0, 0.0, 0.0):
        moved = Transform(Input=source)
        moved.Transform = "Transform"
        moved.Transform.Translate = list(offset)
        source = moved
    # Tube needs polygonal lines: tessellate the curved (Lagrange) cells first.
    lines = ExtractSurface(Input=with_element_index(source))
    lines.NonlinearSubdivisionLevel = 4
    tube = Tube(Input=lines)
    tube.Radius = radius
    tube.NumberofSides = 24
    display = Show(tube, view)
    ColorBy(display, ("CELLS", "element"))
    lut = GetColorTransferFunction("element")
    lut.ApplyPreset("Cool to Warm", True)
    lut.RescaleTransferFunction(1.0, float(num_elements))
    lut.AutomaticRescaleRangeMode = "Never"
    display.SetScalarBarVisibility(view, False)
    endpoints = Show(element_endpoints(name, offset), view)
    endpoints.ColorArrayName = ["POINTS", ""]
    endpoints.AmbientColor = BLACK
    endpoints.DiffuseColor = BLACK
    endpoints.RenderPointsAsSpheres = 1
    endpoints.PointSize = 12
    return display


def set_camera(view, position, focal_point, view_up, zoom=1.0):
    view.CameraPosition = position
    view.CameraFocalPoint = focal_point
    view.CameraViewUp = view_up
    view.ResetCamera(False)
    camera = view.GetActiveCamera()
    if view.CameraParallelProjection:
        view.CameraParallelScale = view.CameraParallelScale / zoom
    else:
        camera.Dolly(zoom)


def save_single(view, filename, size):
    SaveScreenshot(os.path.join(png_dir, filename), view, ImageResolution=size)


def save_pair(view_1, view_2, filename, size, fraction=0.5, vertical=False):
    layout = CreateLayout(filename)
    if vertical:
        layout.SplitVertical(0, fraction)
    else:
        layout.SplitHorizontal(0, fraction)
    AssignViewToLayout(view=view_1, layout=layout, hint=1)
    AssignViewToLayout(view=view_2, layout=layout, hint=2)
    layout.SetSize(*size)
    SaveScreenshot(os.path.join(png_dir, filename), layout, ImageResolution=size)


# ------------------------------------------------------------------------------------------
# 1D: the uniform and non-uniform lines.
view = new_view([1200, 360], parallel=True)
show_curve("line", view, 4, radius=0.02)
show_curve("nonuniform_line", view, 4, offset=(0.0, -0.4, 0.0), radius=0.02)
set_camera(view, [1.0, -0.2, 5.0], [1.0, -0.2, 0.0], [0.0, 1.0, 0.0], zoom=3.2)
save_single(view, "geometry_lines.png", [1200, 360])

# 1D: the parametric line and the spiral.
top = new_view([1200, 300], parallel=True)
show_curve("parametric_line", top, 16, radius=0.08)
show_title("parametric domain", top)
set_camera(top, [2 * 3.14159, 0.0, 10.0], [2 * 3.14159, 0.0, 0.0], [0.0, 1.0, 0.0], 0.92)
bottom = new_view([1200, 900], parallel=True)
show_curve("spiral", bottom, 16, radius=0.025)
show_title("Archimedean spiral", bottom)
set_camera(bottom, [0.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 1.0)
save_pair(top, bottom, "geometry_spiral.png", [1200, 1200], fraction=0.25, vertical=True)

# ------------------------------------------------------------------------------------------
# 2D: the unit square with its element numbers.
view = new_view([800, 800], parallel=True)
show_elements(XMLUnstructuredGridReader(FileName=[vtk_file("square")]), view, 12)
show_wireframe("square", view, width=3.0)
num_x, num_y = 4, 3
for j in range(num_y):
    for i in range(num_x):
        label = a3DText(Text=str(i + j * num_x + 1))
        placed = Transform(Input=label)
        placed.Transform = "Transform"
        placed.Transform.Scale = [0.06, 0.06, 1.0]
        placed.Transform.Translate = [
            (i + 0.5) / num_x - 0.02 * len(str(i + j * num_x + 1)),
            (j + 0.5) / num_y - 0.03,
            0.01,
        ]
        display = Show(placed, view)
        display.ColorArrayName = ["POINTS", ""]
        display.AmbientColor = BLACK
        display.DiffuseColor = BLACK
set_camera(view, [0.5, 0.5, 5.0], [0.5, 0.5, 0.0], [0.0, 1.0, 0.0], zoom=2.6)
save_single(view, "geometry_square.png", [800, 800])

# 2D: the parametric rectangle and the half annulus.
left = new_view([500, 800], parallel=True)
show_elements(XMLUnstructuredGridReader(FileName=[vtk_file("parametric_rectangle")]), left, 48)
show_wireframe("parametric_rectangle", left)
show_title("parametric domain", left)
set_camera(left, [1.5, 1.6, 10.0], [1.5, 1.6, 0.0], [0.0, 1.0, 0.0], zoom=0.85)
right = new_view([1000, 800], parallel=True)
show_elements(XMLUnstructuredGridReader(FileName=[vtk_file("half_annulus")]), right, 48, True)
show_wireframe("half_annulus", right)
show_title("half annulus", right)
set_camera(right, [0.45, 1.0, 10.0], [0.45, 1.0, 0.0], [0.0, 1.0, 0.0], zoom=0.8)
save_pair(left, right, "geometry_half_annulus.png", [1500, 800], fraction=1 / 3)

# 2D: the parametric rectangle and the helicoid.
left = new_view([500, 900], parallel=True)
show_elements(XMLUnstructuredGridReader(FileName=[vtk_file("parametric_rectangle_2pi")]), left, 96)
show_wireframe("parametric_rectangle_2pi", left)
show_title("parametric domain", left)
set_camera(left, [1.5, 3.14, 10.0], [1.5, 3.14, 0.0], [0.0, 1.0, 0.0], zoom=0.42)
right = new_view([1000, 900])
show_elements(XMLUnstructuredGridReader(FileName=[vtk_file("helicoid")]), right, 96, True)
show_wireframe("helicoid", right)
show_title("helicoid", right)
set_camera(right, [6.0, -6.5, 4.5], [0.7, 0.0, 1.0], [0.0, 0.0, 1.0], zoom=1.1)
save_pair(left, right, "geometry_helicoid.png", [1500, 900], fraction=1 / 3)

# ------------------------------------------------------------------------------------------
# 3D: the unit cube.
view = new_view([800, 800])
show_elements(XMLUnstructuredGridReader(FileName=[vtk_file("cube")]), view, 24, True)
show_wireframe("cube", view, width=3.0)
set_camera(view, [3.0, -2.2, 2.4], [0.6, 0.5, 0.5], [0.0, 0.0, 1.0], zoom=0.8)
save_single(view, "geometry_cube.png", [800, 800])

# 3D: the parametric box and the helical duct.
left = new_view([700, 900])
show_elements(XMLUnstructuredGridReader(FileName=[vtk_file("parametric_box")]), left, 144)
show_wireframe("parametric_box", left)
show_title("parametric domain", left)
set_camera(left, [7.0, -2.0, 4.0], [1.5, 4.7, 0.5], [0.0, 0.0, 1.0], zoom=1.0)
right = new_view([1000, 900])
show_elements(XMLUnstructuredGridReader(FileName=[vtk_file("helical_duct")]), right, 144, True)
show_wireframe("helical_duct", right)
show_title("helical duct", right)
set_camera(right, [7.0, -8.0, 6.0], [0.3, 0.0, 2.0], [0.0, 0.0, 1.0], zoom=1.15)
save_pair(left, right, "geometry_helical_duct.png", [1600, 900], fraction=0.4)
