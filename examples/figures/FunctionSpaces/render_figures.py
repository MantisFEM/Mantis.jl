# Renders the ParaView figures of the FunctionSpaces example.
#
# Usage (normally called by make_figures.jl):
#   pvbatch render_figures.py <vtk_dir> <png_dir>
#
# <vtk_dir> must contain the VTK files written by the example.

import os
import sys

from paraview.simple import (
    AssignViewToLayout,
    ColorBy,
    CreateLayout,
    CreateRenderView,
    ExtractSubset,
    GetColorTransferFunction,
    GetScalarBar,
    ResampleToImage,
    SaveScreenshot,
    Show,
    Text,
    XMLUnstructuredGridReader,
)

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


def color_by_basis_function(display, view, maximum, scalar_bar=False):
    # The VTK files store the value of the basis function in the point array "B".
    ColorBy(display, ("POINTS", "B"))
    lut = GetColorTransferFunction("B")
    lut.ApplyPreset("Viridis", True)
    lut.RescaleTransferFunction(0.0, maximum)
    lut.AutomaticRescaleRangeMode = "Never"
    display.SetScalarBarVisibility(view, scalar_bar)
    if scalar_bar:
        bar = GetScalarBar(lut, view)
        bar.Title = "basis function"
        bar.ComponentTitle = ""
        bar.TitleColor = BLACK
        bar.LabelColor = BLACK
        bar.TitleFontSize = FONT_SIZE
        bar.LabelFontSize = FONT_SIZE - 4
        bar.AddRangeLabels = 0
        bar.WindowLocation = "Any Location"
        bar.Orientation = "Vertical"
        bar.Position = [0.88, 0.15]
        bar.ScalarBarLength = 0.7


def show_basis_function(source, view, maximum, scalar_bar=False):
    display = Show(source, view)
    display.SetRepresentationType("Surface")  # The default for 3D image data is "Outline".
    display.NonlinearSubdivisionLevel = 3
    display.Ambient = 0.25
    display.Diffuse = 0.85
    color_by_basis_function(display, view, maximum, scalar_bar)
    return display


def show_wireframe(source, view, width=2.0):
    display = Show(source, view)
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


# The screenshots are saved without transparency: with it, some pixels of the surfaces come
# out partially transparent.
def save_single(view, filename, size):
    SaveScreenshot(
        os.path.join(png_dir, filename), view, ImageResolution=size, TransparentBackground=0
    )


def save_pair(view_1, view_2, filename, size, fraction=0.5):
    layout = CreateLayout(filename)
    layout.SplitHorizontal(0, fraction)
    AssignViewToLayout(view=view_1, layout=layout, hint=1)
    AssignViewToLayout(view=view_2, layout=layout, hint=2)
    layout.SetSize(*size)
    SaveScreenshot(
        os.path.join(png_dir, filename), layout, ImageResolution=size, TransparentBackground=0
    )


def cut_open(source, num_samples=97):
    # The (Lagrange) cells are sampled on a fine uniform grid, so that the volume shows the
    # high-order field without facets. Only the samples with x_2 >= 1/2 are kept, which cuts
    # the cube open.
    sampled = ResampleToImage(Input=source)
    sampled.SamplingDimensions = [num_samples] * 3
    half = ExtractSubset(Input=sampled)
    last = num_samples - 1
    half.VOI = [0, last, last // 2, last, 0, last]
    return half


# ------------------------------------------------------------------------------------------
# 3D: a basis function on the cube, cut open along the plane x_2 = 1/2. The element edges
# of the whole cube are drawn, so that the removed half remains visible.
view = new_view([900, 800])
cube = XMLUnstructuredGridReader(FileName=[vtk_file("cube_basis_45-B")])
show_basis_function(cut_open(cube), view, 0.37, True)
edges = XMLUnstructuredGridReader(FileName=[vtk_file("cube_basis_45-B_wireframe")])
show_wireframe(edges, view, width=1.5)
set_camera(view, [2.0, -2.4, 2.2], [0.5, 0.5, 0.45], [0.0, 0.0, 1.0], zoom=1.0)
save_single(view, "function_spaces_cube.png", [900, 800])

# ------------------------------------------------------------------------------------------
# 2D: a basis function on the parametric rectangle and on the helicoid.
left = new_view([500, 900], parallel=True)
rectangle = XMLUnstructuredGridReader(FileName=[vtk_file("helicoid_rectangle_basis-B")])
show_basis_function(rectangle, left, 0.57)
show_wireframe(
    XMLUnstructuredGridReader(FileName=[vtk_file("helicoid_rectangle_basis-B_wireframe")]),
    left,
)
show_title("parametric domain", left)
set_camera(left, [1.5, 3.14, 10.0], [1.5, 3.14, 0.0], [0.0, 1.0, 0.0], zoom=0.42)
right = new_view([1000, 900])
helicoid = XMLUnstructuredGridReader(FileName=[vtk_file("helicoid_basis-B")])
show_basis_function(helicoid, right, 0.57, True)
show_wireframe(
    XMLUnstructuredGridReader(FileName=[vtk_file("helicoid_basis-B_wireframe")]), right
)
show_title("helicoid", right)
set_camera(right, [6.0, -6.5, 4.5], [0.7, 0.0, 1.0], [0.0, 0.0, 1.0], zoom=1.1)
save_pair(left, right, "function_spaces_helicoid.png", [1500, 900], fraction=1 / 3)
