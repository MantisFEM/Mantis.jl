using Documenter
using DocumenterCitations
using Mantis

Manual = [
    "Manual/InstallGuide.md",
]

Tutorials = [
    "Documentation" => [
        "Tutorials/CreatingDocs.md",
        "Tutorials/CreatingDocsPage.md",
    ],
    "Running MANTIS" => [
        "Tutorials/Test.md",
    ]
]

DevelDocs = [
    "DevelDocs/MainPageDevelDocs.md",
    "Documents" => [
        "DevelDocs/Documentation.md",
    ],
]


Pages = [
    "index.md",
    "Manual" => Manual,
    "Tutorials" => Tutorials,
    "Developer Documentation" => DevelDocs,
]


# We set the LaTeX engine to be the (non-default) MathJax3 engine. This 
# allows for a more straightforward specification of the latex packages 
# and has a more consistent (with surrounding text) style for inline 
# math. The default options for MathJax will do for now.
math_engine = Documenter.MathJax3()

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
)


# The modules option will raise an error when some docstrings from the 
# listed modules are not included in the docs. Due to an issue in Julia 
# (see issue #45174) this does not always go well with functors 
# (callable structs) so the docstrings should be moved to the type 
# definitions as work-around.
# Author names are ordered alphabetically on last name.
makedocs(
    modules  = [Mantis.FunctionSpaces, Mantis.Quadrature], 
    format   = format_setup,
    sitename = "MANTIS.jl",
    authors  = "Diogo Costa Cabanas, Joey Dekker, Artur Palha, Deepesh Toshniwal",
    pages    = Pages,
    plugins  = [bib],
)
