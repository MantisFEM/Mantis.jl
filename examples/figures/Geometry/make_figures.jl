# Generates the figures of the Geometry example.
#
# The example is run to write its VTK files, which are then rendered with ParaView. The
# diagram of the maps is drawn by make_maps_diagram.py. The resulting PNG and SVG files are
# stored in docs/src/assets/Examples/Geometry.
#
# Usage, from the Mantis.jl folder (ParaView's `pvbatch` and `python3` must be available):
#   julia --project=docs examples/figures/Geometry/make_figures.jl [path/to/pvbatch]

const mantis_dir = normpath(joinpath(@__DIR__, "..", "..", ".."))
const examples_dir = joinpath(mantis_dir, "examples", "src")
const png_dir = joinpath(mantis_dir, "docs", "src", "assets", "Examples", "Geometry")
const pvbatch = isempty(ARGS) ? "pvbatch" : ARGS[1]

# Run the example in its own module and collect the VTK files from its `output_dir`.
@info "Running the Geometry example"
example_module = Module(:GeometryExample)
Core.eval(example_module, :(include(path) = Base.include($example_module, path)))
Base.include(example_module, joinpath(examples_dir, "Geometry.jl"))
vtk_dir = Core.eval(example_module, :output_dir)

@info "Rendering figures into $png_dir"
run(`$pvbatch $(joinpath(@__DIR__, "render_figures.py")) $vtk_dir $png_dir`)
diagram_script = joinpath(@__DIR__, "make_maps_diagram.py")
run(`python3 $diagram_script $(joinpath(png_dir, "geometry_maps.svg"))`)
