# Draws the diagram of the canonical element, the parametric domain and the manifold, with
# the maps between them, used in the Geometry example.
#
# Usage, from any folder:
#   python3 make_maps_diagram.py <output.svg>

import math
import sys

output = sys.argv[1]

INK = "#1a1a1a"
GRID = "#555555"
FILL = "#f4a582"
FILL_DARK = "#d6604d"
SURFACE = "#e8eef7"
ARROW = "#2166ac"

parts = []


def add(element):
    parts.append(element)


def path(points, closed=False, **attributes):
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    if closed:
        d += " Z"
    style = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attributes.items())
    add(f'<path d="{d}" {style}/>')


def text(x, y, content, size=20, anchor="middle", italic=True, fill=INK, **attributes):
    style = "font-style:italic;" if italic else ""
    extra = " ".join(f'{key.replace("_", "-")}="{value}"' for key, value in attributes.items())
    add(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
        f'font-family="Times New Roman, Times, serif" style="{style}" fill="{fill}" '
        f"{extra}>{content}</text>"
    )


def arrow(points, label=None, label_position=None):
    path(points, stroke=ARROW, stroke_width=2.5, fill="none", marker_end="url(#arrow)")
    if label is not None:
        text(*label_position, label, size=22, fill=ARROW)


def omega_hat(x, y, size=22):
    # A capital omega with a hat on top.
    text(x, y, "Ω", size=size, italic=False)
    text(x, y - 0.62 * size, "^", size=0.8 * size, italic=False)


# ------------------------------------------------------------------------------------------
# Canonical element [0, 1]^2.
x0, y0, side = 55, 245, 120
path([(x0, y0), (x0 + side, y0), (x0 + side, y0 - side), (x0, y0 - side)], True,
     fill=FILL, stroke=INK, stroke_width=2)
path([(x0, y0), (x0 + side + 25, y0)], stroke=INK, stroke_width=1.2, marker_end="url(#axis)")
path([(x0, y0), (x0, y0 - side - 25)], stroke=INK, stroke_width=1.2, marker_end="url(#axis)")
text(x0 + side + 30, y0 + 22, "ξ<tspan baseline-shift='sub' font-size='13'>1</tspan>", 18)
text(x0 - 18, y0 - side - 22, "ξ<tspan baseline-shift='sub' font-size='13'>2</tspan>", 18)
text(x0 - 8, y0 + 20, "0", 15, italic=False)
text(x0 + side, y0 + 20, "1", 15, italic=False)
text(x0 - 14, y0 - side + 5, "1", 15, italic=False)
text(x0 + side / 2, 60, "canonical element", 17, italic=False)
text(x0 + side / 2, 300, "[0, 1]<tspan baseline-shift='super' font-size='13'>n</tspan>", 20,
     italic=False)

# ------------------------------------------------------------------------------------------
# Parametric domain: a rectangle with 4 x 3 elements, element e highlighted.
px0, py0, width, height = 305, 255, 200, 150
nx, ny = 4, 3
ex, ey = 1, 1  # zero-based indices of the highlighted element
dx, dy = width / nx, height / ny
path([(px0 + ex * dx, py0 - ey * dy), (px0 + (ex + 1) * dx, py0 - ey * dy),
      (px0 + (ex + 1) * dx, py0 - (ey + 1) * dy), (px0 + ex * dx, py0 - (ey + 1) * dy)],
     True, fill=FILL, stroke="none")
for i in range(nx + 1):
    path([(px0 + i * dx, py0), (px0 + i * dx, py0 - height)], stroke=GRID, stroke_width=1.2)
for j in range(ny + 1):
    path([(px0, py0 - j * dy), (px0 + width, py0 - j * dy)], stroke=GRID, stroke_width=1.2)
path([(px0, py0), (px0 + width, py0), (px0 + width, py0 - height), (px0, py0 - height)], True,
     fill="none", stroke=INK, stroke_width=2)
text(px0 + (ex + 0.5) * dx, py0 - (ey + 0.5) * dy + 7, "Ω", 18, italic=False)
text(px0 + (ex + 0.5) * dx, py0 - (ey + 0.5) * dy - 6, "^", 14, italic=False)
text(px0 + (ex + 0.5) * dx + 13, py0 - (ey + 0.5) * dy + 12, "e", 13)
text(px0 + width / 2, 60, "parametric domain", 17, italic=False)
omega_hat(px0 + width / 2 - 25, 300)
text(px0 + width / 2 + 15, 300, "⊂ ℝ<tspan baseline-shift='super' font-size='13'>n</tspan>",
     20, italic=False)

# ------------------------------------------------------------------------------------------
# Manifold: a curved surface in 3D, seen in an oblique projection.
def surface(u, v):
    # A gently curved surface patch over [0, 1]^2.
    x = u
    y = v
    z = 0.3 * math.sin(math.pi * u) * (1.0 - 0.6 * v) + 0.25 * v * v
    return x, y, z


def project(point):
    x, y, z = point
    sx = 655 + 165 * x + 55 * y
    sy = 262 - 10 * x - 125 * y - 75 * z
    return sx, sy


def curve(u_range, v_range, samples=40):
    points = []
    for k in range(samples + 1):
        s = k / samples
        u = u_range[0] + s * (u_range[1] - u_range[0])
        v = v_range[0] + s * (v_range[1] - v_range[0])
        points.append(project(surface(u, v)))
    return points


outline = (curve((0, 1), (0, 0)) + curve((1, 1), (0, 1)) + curve((1, 0), (1, 1))
           + curve((0, 0), (1, 0)))
path(outline, True, fill=SURFACE, stroke="none")
element = (curve((ex / nx, (ex + 1) / nx), (ey / ny, ey / ny))
           + curve(((ex + 1) / nx, (ex + 1) / nx), (ey / ny, (ey + 1) / ny))
           + curve(((ex + 1) / nx, ex / nx), ((ey + 1) / ny, (ey + 1) / ny))
           + curve((ex / nx, ex / nx), ((ey + 1) / ny, ey / ny)))
path(element, True, fill=FILL, stroke="none")
for i in range(nx + 1):
    path(curve((i / nx, i / nx), (0, 1)), stroke=GRID, stroke_width=1.2, fill="none")
for j in range(ny + 1):
    path(curve((0, 1), (j / ny, j / ny)), stroke=GRID, stroke_width=1.2, fill="none")
path(outline, True, fill="none", stroke=INK, stroke_width=2)
cx, cy = project(surface((ex + 0.5) / nx, (ey + 0.5) / ny))
text(cx, cy + 7, "ℳ<tspan baseline-shift='sub' font-size='13'>e</tspan>", 18)
text(775, 60, "manifold", 17, italic=False)
text(780, 300, "ℳ = Φ(", 20, anchor="end")
omega_hat(791, 300)
text(802, 300, ") ⊂ ℝ<tspan baseline-shift='super' font-size='13'>m</tspan>", 20,
     anchor="start", italic=False)

# ------------------------------------------------------------------------------------------
# The maps.
arrow([(195, 170), (285, 170)], "φ<tspan baseline-shift='sub' font-size='15'>e</tspan>",
      (240, 155))
arrow([(520, 170), (630, 170)], "Φ", (575, 155))
composite = []
for k in range(41):
    s = k / 40
    x = 115 + s * (770 - 115)
    y = 330 + 55 * math.sin(math.pi * s)
    composite.append((x, y))
arrow(composite, "Φ<tspan baseline-shift='sub' font-size='15'>e</tspan> = Φ ∘ "
      "φ<tspan baseline-shift='sub' font-size='15'>e</tspan>", (437, 412))

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 430" width="900" height="430">
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" fill="{ARROW}"/>
</marker>
<marker id="axis" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M 0 0 L 10 5 L 0 10 z" fill="{INK}"/>
</marker>
</defs>
<rect x="0" y="0" width="900" height="430" rx="12" fill="#ffffff"/>
{chr(10).join(parts)}
</svg>
"""

with open(output, "w") as file:
    file.write(svg)
