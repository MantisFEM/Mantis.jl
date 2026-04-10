using Documenter
using DocumenterCitations
using Mantis
using Literate
using DocumenterVitepress

# Generate the notebooks and example pages based on the .jl files in Mantis/examples/src.
# This generation ensures that the examples are up-to-date with the latest version of
# MANTIS.
mantis_dir = dirname(dirname(pathof(Mantis)))
examples_dir = joinpath(mantis_dir, "examples", "src")

example_names = String[]
for example in readdir(examples_dir)
    if endswith(example, ".jl")
        path_to_example = joinpath(examples_dir, example)

        push!(example_names, example[1:(end - 3)])  # Remove the file extension from the name.

        # Literate.notebook(path_to_example, joinpath(mantis_dir, "examples", "notebooks"))

        Literate.markdown(
            path_to_example,
            joinpath(mantis_dir, "docs", "src", "Examples");
            flavor=Literate.DocumenterFlavor(),
        )
    end
end

Examples = [
    joinpath("Examples", "Introduction.md"),
    map(example_name -> joinpath("Examples", "$example_name.md"), example_names)...,
]

Design = [
    joinpath("Design", "DesignIntroduction.md"),
    "Philosophy" => [joinpath("Design", "Philosophy", "WhyMantis.md")],
    "Theory" => joinpath.("Design", "Theory", ["FEM.md", "DifferentialForms.md"]),
    "Modules" =>
        joinpath.(
            "Design",
            "Modules",
            [
                "Analysis.md",
                "Assemblers.md",
                "Forms.md",
                "FunctionSpaces.md",
                "GeneralHelpers.md",
				"Hierarchy.md",
                "Geometry.md",
                "Mesh.md",
                "Plot.md",
                "Points.md",
                "Quadrature.md",
                "TimeIntegrators.md",
            ],
        ),
]

Support = [
    "Getting help" => joinpath("Support", "GettingHelp.md"),
    "Submitting a bug report" => joinpath("Support", "SubmitBugReport.md"),
    "Contributing" => joinpath("Support", "Contributing.md"),
    "Requesting a new feature" => joinpath("Support", "FeatureRequest.md"),
]

ReleaseHistory = [
    "v0.6 Acanthops onorei" => joinpath("ReleaseHistory", "v0-Acanthops", "v0.6-onorei.md"),
    "v0.5 Acanthops godmani" =>
        joinpath("ReleaseHistory", "v0-Acanthops", "v0.5-godmani.md"),
    "v0.4 Acanthops falcata" =>
        joinpath("ReleaseHistory", "v0-Acanthops", "v0.4-falcata.md"),
    "v0.3 Acanthops erosa" => joinpath("ReleaseHistory", "v0-Acanthops", "v0.3-erosa.md"),
    "v0.2 Acanthops centralis" =>
        joinpath("ReleaseHistory", "v0-Acanthops", "v0.2-centralis.md"),
    "v0.1 Acanthops brunneri" =>
        joinpath("ReleaseHistory", "v0-Acanthops", "v0.1-brunneri.md"),
]

Pages = [
    "Getting Started" => "GettingStarted.md",
    "Examples" => Examples,
    "Design" => Design,
    "Support(ing)" => Support,
    "Releases" => ReleaseHistory,
    "Bibliography" => [
        joinpath("BibliographicInformation", "references.md"),
        joinpath("BibliographicInformation", "HowToCite.md"),
    ],
]

# References are handled by DocumenterCitations so this should be set up.
bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)

# The modules option will raise an error when some docstrings from the
# listed modules are not included in the docs. Due to an issue in Julia
# (see issue #45174) this does not always go well with functors
# (callable structs) so the docstrings should be moved to the type
# definitions as work-around.
# Author names are ordered alphabetically on last name.
makedocs(;
    modules=[
        Mantis.Analysis,
        Mantis.Assemblers,
        Mantis.Forms,
        Mantis.FunctionSpaces,
        Mantis.GeneralHelpers,
        Mantis.Geometry,
        Mantis.Hierarchy,
        Mantis.Mesh,
        Mantis.Plot,
        Mantis.Points,
        Mantis.Quadrature,
        Mantis.TimeIntegrators,
    ],
    repo=Remotes.GitHub("MantisFEM", "Mantis.jl"),
    sitename="Mantis.jl",
    authors="Diogo C. Cabanas, Joey Dekker, Artur Palha, Deepesh Toshniwal",
    pages=Pages,
    plugins=[bib],
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/MantisFEM/Mantis.jl", devbranch = "dev", devurl = "dev"
    ),
)
DocumenterVitepress.deploydocs(;
    repo="github.com/MantisFEM/Mantis.jl",
    target="build", # this is where Vitepress stores its output
    devbranch="dev",
    devurl = "dev",
    branch="gh-pages",
    push_preview=true,
)
