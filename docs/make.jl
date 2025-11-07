using Documenter
using DocumenterCitations
using Mantis
using Literate

# Generate the notebooks and example pages based on the .jl files in Mantis/examples/src.
# This generation ensures that the examples are up-to-date with the latest version of
# MANTIS.
mantis_dir = dirname(dirname(pathof(Mantis)))
examples_dir = joinpath(mantis_dir, "examples", "src")

example_names = String[]
for example in readdir(examples_dir)

    if endswith(example, ".jl")
        path_to_example = joinpath(examples_dir, example)

        push!(example_names, example[1:end-3])  # Remove the file extension from the name.

        Literate.notebook(
            path_to_example,
            joinpath(mantis_dir, "examples", "notebooks"),
        )

        Literate.markdown(
            path_to_example,
            joinpath(mantis_dir, "docs", "src", "Examples"),
            flavor = Literate.DocumenterFlavor(),
        )
    end
end


Manual = [
    "Manual/InstallGuide.md",
]

Examples = [
    "Examples/Introduction.md",
    map(example_name -> "Examples/$example_name.md", example_names)...,
]

Tutorials = [
    "Documentation" => [
        "Tutorials/BuildingDocs.md",
        "Tutorials/CreatingDocsPage.md",
    ],
    "Running MANTIS" => [
        "Tutorials/RunningMantis.md",
    ]
]

DevelDocs = [
    "DevelDocs/MainPageDevelDocs.md",
    "Documents" => [
        "DevelDocs/Documentation.md",
    ],
    "Modules" => [
        "DevelDocs/Modules/Analysis.md",
        "DevelDocs/Modules/Assemblers.md",
        "DevelDocs/Modules/Forms.md",
        "DevelDocs/Modules/FunctionSpaces.md",
        "DevelDocs/Modules/GeneralHelpers.md",
        "DevelDocs/Modules/Geometry.md",
        "DevelDocs/Modules/Mesh.md",
        "DevelDocs/Modules/Plot.md",
        "DevelDocs/Modules/Points.md",
        "DevelDocs/Modules/Quadrature.md",
    ],
]


Pages = [
    "index.md",
    "Manual" => Manual,
    "Examples" => Examples,
    "Tutorials" => Tutorials,
    "Developer Documentation" => DevelDocs,
]


# We set the LaTeX engine to be the (non-default) MathJax3 engine. This
# allows for a more straightforward specification of the latex packages
# and has a more consistent (with surrounding text) style for inline
# math. The default options for MathJax will do for now.
math_engine = Documenter.MathJax3(Dict(
    :tex => Dict(
        "inlineMath" => [["\$","\$"], ["\\(","\\)"]],
        "tags" => "ams",
        "packages" => ["base", "ams", "autoload"])
))

# References are handled by DocumenterCitations so this should be set up.
bib = CitationBibliography(
    joinpath(@__DIR__, "src", "refs.bib");
    style=:numeric
)

# Update the formatting to include the new math engine. Also make sure
# that the favicon is found (the small logo in the tab bar).
format_setup = Documenter.HTML(
    assets = [
        "assets/favicon.ico"
        "assets/citations.css"
    ],
    mathengine=math_engine,
    size_threshold = nothing, # Prevents errors for large HTML files. Temporary only.
)


# The modules option will raise an error when some docstrings from the
# listed modules are not included in the docs. Due to an issue in Julia
# (see issue #45174) this does not always go well with functors
# (callable structs) so the docstrings should be moved to the type
# definitions as work-around.
# Author names are ordered alphabetically on last name.
makedocs(
    modules  = [Mantis.Assemblers, Mantis.FunctionSpaces, Mantis.Quadrature],
    format   = format_setup,
    sitename = "MANTIS.jl",
    authors  = "Diogo Costa Cabanas, Joey Dekker, Artur Palha, Deepesh Toshniwal",
    pages    = Pages,
    plugins  = [bib],
)
