# Renders the figures of the Forms example with ParaView.
#
# Usage (normally called by make_figures.jl):
#   pvbatch render_figures.py <vtk_dir> <png_dir>
#
# <vtk_dir> must contain the VTK files written by the example.

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from paraview.simple import (
    AssignViewToLayout,
    ColorBy,
    CreateLayout,
    CreateRenderView,
    GetColorTransferFunction,
    GetScalarBar,
    Glyph,
    MaskPoints,
    Outline,
    ResampleToImage,
    SaveScreenshot,
    Show,
    Text,
    Threshold,
    XMLUnstructuredGridReader,
)
import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader

vtk_dir, png_dir = sys.argv[1], sys.argv[2]
os.makedirs(png_dir, exist_ok=True)

WHITE = [1.0, 1.0, 1.0]
BLACK = [0.0, 0.0, 0.0]
EDGE_COLOR = [0.15, 0.15, 0.15]
FONT_SIZE = 22
PANEL_SIZE = [600, 560]


def vtk_file(name):
    return os.path.join(vtk_dir, name + ".vtu")


def read_grid(name):
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtk_file(name))
    reader.Update()
    return reader.GetOutput()


def array_name(name):
    # Each file stores a single point array, named after the label of the form.
    return read_grid(name).GetPointData().GetArrayName(0)


def new_view():
    view = CreateRenderView()
    view.ViewSize = PANEL_SIZE
    view.OrientationAxesVisibility = 0
    try:
        view.UseColorPaletteForBackground = 0
    except AttributeError:
        pass
    view.Background = WHITE
    return view


def show_title(text, view):
    display = Show(Text(Text=text), view)
    display.WindowLocation = "Upper Center"
    display.FontSize = FONT_SIZE
    display.Color = BLACK


def show_edges(name, view, color=EDGE_COLOR, width=1.5):
    display = Show(XMLUnstructuredGridReader(FileName=[vtk_file(name + "_wireframe")]), view)
    display.ColorArrayName = ["POINTS", ""]
    display.AmbientColor = color
    display.DiffuseColor = color
    display.LineWidth = width


def show_outline(source, view):
    display = Show(Outline(Input=source), view)
    display.ColorArrayName = ["POINTS", ""]
    display.AmbientColor = EDGE_COLOR
    display.DiffuseColor = EDGE_COLOR
    display.LineWidth = 1.5


_num_color_maps = [0]


def set_colors(array, view, display, component=None, color_range=None, title=""):
    # Each panel gets its own, uniquely named color map. ParaView would otherwise share
    # color maps between arrays with the same name, and its names drop the non-ASCII
    # characters of the labels, so that, for example, "f⁰" and "f¹" would collide.
    if component is None:
        ColorBy(display, ("POINTS", array))
    else:
        ColorBy(display, ("POINTS", array, component))
    _num_color_maps[0] += 1
    lut = GetColorTransferFunction("panel_%d" % _num_color_maps[0])
    if component is not None:
        lut.VectorMode = "Magnitude"
    display.LookupTable = lut
    lut.ApplyPreset("Viridis (matplotlib)", True)
    if color_range is None:
        display.RescaleTransferFunctionToDataRange(True, False)
    else:
        lut.AutomaticRescaleRangeMode = "Never"
        lut.RescaleTransferFunction(*color_range)
    display.SetScalarBarVisibility(view, True)
    bar = GetScalarBar(lut, view)
    bar.Title = title
    bar.ComponentTitle = ""
    bar.TitleColor = BLACK
    bar.LabelColor = BLACK
    bar.TitleFontSize = FONT_SIZE - 4
    bar.LabelFontSize = FONT_SIZE - 6
    bar.WindowLocation = "Any Location"
    bar.Orientation = "Vertical"
    bar.Position = [0.88, 0.15]
    bar.ScalarBarLength = 0.6
    bar.AutomaticLabelFormat = 0
    bar.LabelFormat = "%-#.1f"
    bar.RangeLabelFormat = "%-#.1f"


def scalar_panel(name, title, color_range=None):
    view = new_view()
    display = Show(XMLUnstructuredGridReader(FileName=[vtk_file(name)]), view)
    display.NonlinearSubdivisionLevel = 2
    set_colors(array_name(name), view, display, color_range=color_range)
    show_edges(name, view, color=[0.3, 0.3, 0.3], width=1.0)
    show_title(title, view)
    return view


def vector_panel(name, title, num_arrows=15, edges=True):
    # Arrows on a regular grid of num_arrows points per direction, restricted to the domain,
    # scaled so that the longest arrow is about the distance between arrows.
    view = new_view()
    source = XMLUnstructuredGridReader(FileName=[vtk_file(name)])
    array = array_name(name)
    grid = read_grid(name)
    bounds = np.array(grid.GetBounds()).reshape(3, 2)
    extent = bounds[:, 1] - bounds[:, 0]
    dims = [num_arrows if e > 1e-12 else 1 for e in extent]
    spacing = min(e / (n - 1) for e, n in zip(extent, dims) if n > 1)
    vectors = vtk_to_numpy(grid.GetPointData().GetArray(0))
    max_length = np.max(np.linalg.norm(vectors, axis=1))
    if edges:
        show_edges(name, view, color=[0.6, 0.6, 0.6], width=1.0)
    else:
        show_outline(source, view)
    resampled = ResampleToImage(Input=source)
    resampled.UseInputBounds = 1
    resampled.SamplingDimensions = dims
    inside = Threshold(Input=resampled)
    inside.Scalars = ["POINTS", "vtkValidPointMask"]
    inside.LowerThreshold = 0.5
    inside.UpperThreshold = 1.5
    arrows = Glyph(Input=inside, GlyphType="Arrow")
    arrows.OrientationArray = ["POINTS", array]
    arrows.ScaleArray = ["POINTS", array]
    arrows.ScaleFactor = 0.9 * spacing / max_length
    arrows.GlyphMode = "All Points"
    arrows.GlyphType.TipResolution = 16
    arrows.GlyphType.TipRadius = 0.18
    arrows.GlyphType.TipLength = 0.35
    arrows.GlyphType.ShaftResolution = 16
    arrows.GlyphType.ShaftRadius = 0.06
    display = Show(arrows, view)
    set_colors(array, view, display, component="Magnitude", title="length")
    show_title(title, view)
    return view


def top_camera(view, center, half_height):
    view.CameraParallelProjection = 1
    view.CameraPosition = [center[0], center[1], 10.0]
    view.CameraFocalPoint = [center[0], center[1], 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    view.ResetCamera(False)
    view.CameraParallelScale = half_height


def cube_camera(view):
    view.CameraPosition = [3.2, -2.4, 2.6]
    view.CameraFocalPoint = [0.62, 0.42, 0.45]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.ResetCamera(False)
    view.GetActiveCamera().Dolly(0.95)


def save_row(views, filename):
    layout = CreateLayout(filename)
    cells = []
    cell = 0
    for i in range(len(views) - 1):
        layout.SplitHorizontal(cell, 1.0 / (len(views) - i))
        cells.append(2 * cell + 1)
        cell = 2 * cell + 2
    cells.append(cell)
    for view, cell in zip(views, cells):
        AssignViewToLayout(view=view, layout=layout, hint=cell)
    size = [PANEL_SIZE[0] * len(views), PANEL_SIZE[1]]
    layout.SetSize(*size)
    SaveScreenshot(os.path.join(png_dir, filename), layout, ImageResolution=size)


def save_grid(views, filename):
    # A 2 × 2 grid: views[0] and views[1] on top, views[2] and views[3] below.
    layout = CreateLayout(filename)
    layout.SplitVertical(0, 0.5)
    layout.SplitHorizontal(1, 0.5)
    layout.SplitHorizontal(2, 0.5)
    for view, cell in zip(views, (3, 4, 5, 6)):
        AssignViewToLayout(view=view, layout=layout, hint=cell)
    size = [2 * PANEL_SIZE[0], 2 * PANEL_SIZE[1]]
    layout.SetSize(*size)
    SaveScreenshot(os.path.join(png_dir, filename), layout, ImageResolution=size)


def line_data(name):
    grid = read_grid(name)
    x = vtk_to_numpy(grid.GetPoints().GetData())[:, 0]
    values = vtk_to_numpy(grid.GetPointData().GetArray(0)).reshape(len(x), -1)[:, 0]
    order = x.argsort()
    return x[order], values[order]


# ------------------------------------------------------------------------------------------
# 1D: the value of u⁰ and the density of v¹.
fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
for name, label in (("forms_1D-u⁰", "$u^0_h$"), ("forms_1D-v¹", "density of $v^1_h$")):
    x, values = line_data(name)
    ax.plot(x, values, label=label)
for breakpoint in (0.0, 0.5, 1.0, 1.5, 2.0):
    ax.axvline(breakpoint, color="0.85", linewidth=0.8, zorder=0)
ax.set_xlabel("$x$")
ax.set_ylim(0.0, 2.1)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(png_dir, "forms_1D.png"))

# ------------------------------------------------------------------------------------------
# 2D: u⁰, the vector proxy of v¹ and the density of w².
views = [
    scalar_panel("forms_2D-u⁰", "$u^0_h$"),
    vector_panel("forms_2D-v¹", "$v^1_h$", num_arrows=13),
    scalar_panel("forms_2D-w²", "$\\star w^2_h$"),
]
for view in views:
    top_camera(view, (0.66, 0.5), 0.68)
save_row(views, "forms_2D.png")

# ------------------------------------------------------------------------------------------
# 3D: u⁰, the vector proxies of v¹ and ★w², and the density of ρ³.
views = [
    scalar_panel("forms_3D-u⁰", "$u^0_h$"),
    vector_panel("forms_3D-v¹", "$v^1_h$", num_arrows=6, edges=False),
    vector_panel("forms_3D-star(w²)", "$\\star w^2_h$", num_arrows=6, edges=False),
    scalar_panel("forms_3D-ρ³", "$\\star \\rho^3_h$"),
]
for view in views:
    cube_camera(view)
save_row(views, "forms_3D.png")

# ------------------------------------------------------------------------------------------
# L2 projection on the unit square: exact, coarse and fine.
views = [
    scalar_panel("projection_exact-f⁰", "exact", color_range=(0.0, 1.0)),
    scalar_panel("projection_coarse-Λ⁰", "coarse", color_range=(0.0, 1.0)),
    scalar_panel("projection_fine-Λ⁰", "fine", color_range=(0.0, 1.0)),
]
for view in views:
    top_camera(view, (0.66, 0.5), 0.68)
save_row(views, "projection_f0.png")

views = [
    vector_panel("projection_exact-f¹", "exact"),
    vector_panel("projection_coarse-Λ¹", "coarse"),
    vector_panel("projection_fine-Λ¹", "fine"),
]
for view in views:
    top_camera(view, (0.66, 0.5), 0.68)
save_row(views, "projection_f1.png")

# ------------------------------------------------------------------------------------------
# L2 projection on the half annulus.
views = [
    scalar_panel("annulus_exact-f⁰", "$f^0$", color_range=(-1.0, 1.0)),
    scalar_panel("annulus_projection-Λ⁰", "$f^0_h$", color_range=(-1.0, 1.0)),
    vector_panel("annulus_exact-f¹", "$f^1$", num_arrows=18),
    vector_panel("annulus_projection-Λ¹", "$f^1_h$", num_arrows=18),
]
for view in views:
    top_camera(view, (0.4, 0.95), 1.95)
save_grid(views, "projection_annulus.png")

# ------------------------------------------------------------------------------------------
# Operations on the fine projections.
views = [
    vector_panel("operations-d(Λ⁰)", "$d f^0_h$"),
    scalar_panel("operations-d(Λ¹)", "$\\star d f^1_h$"),
    scalar_panel("operations-(Λ¹wedged(Λ⁰))", "$\\star(f^1_h \\wedge d f^0_h)$"),
]
for view in views:
    top_camera(view, (0.66, 0.5), 0.68)
save_row(views, "operations.png")
