# Generates the ParaView figures of the FunctionSpaces example.
#
# The example is run to write its VTK files, which are then rendered with ParaView. The
# figures drawn with `Makie` are created when the documentation is built. The resulting PNG
# files are stored in docs/src/assets/Examples/FunctionSpaces.
#
# Usage, from the Mantis.jl folder (ParaView's `pvbatch` must be available):
#   julia --project=docs examples/figures/FunctionSpaces/make_figures.jl [path/to/pvbatch]

const mantis_dir = normpath(joinpath(@__DIR__, "..", "..", ".."))
const examples_dir = joinpath(mantis_dir, "examples", "src")
const png_dir = joinpath(mantis_dir, "docs", "src", "assets", "Examples", "FunctionSpaces")
const pvbatch = isempty(ARGS) ? "pvbatch" : ARGS[1]

# Run the example in its own module and collect the VTK files from its `output_dir`.
@info "Running the FunctionSpaces example"
example_module = Module(:FunctionSpacesExample)
Core.eval(example_module, :(include(path) = Base.include($example_module, path)))
Base.include(example_module, joinpath(examples_dir, "FunctionSpaces.jl"))
vtk_dir = Core.eval(example_module, :output_dir)

@info "Rendering figures into $png_dir"
run(`$pvbatch $(joinpath(@__DIR__, "render_figures.py")) $vtk_dir $png_dir`)
