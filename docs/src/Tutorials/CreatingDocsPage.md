# How to create a page in the MANTIS docs.
To create a new page, create a `filename.md`-file somewhere in the
`docs` folder (or any of its subfolders). You can type whatever you want
here, which will be rendered into the content of your new page. These
pages are written using `MarkDown` syntax, hence the `.md` extension.
For the supported markdown syntax, see the [Julia MarkDown Docs](https://docs.julialang.org/en/v1/stdlib/Markdown/).
Also check out the [Documenter Showcase](https://documenter.juliadocs.org/stable/showcase/)
as well as the rest of the `Documenter` docs, to look up the syntax for
the special boxes and whatnots.

!!! tip "You can make Admonitions boxes"
    These can highlight some of the important stuff. The colour of the boxes will change
    depending on the type of Admonition.

!!! compat "Requires at least Julia 1.10"
    Mantis was build assuming at least Julia 1.10.

!!! warning "This above may be wrong"
    I did not check that, but it may be useful to include such things.

You can also have a look at some of the pages I already created, which I
also did to test things out. Note that the `Documenter` docs do not
always explain every detail, so some googling may be required.

## Some notes on MarkDown in VScode.
VScode does have MarkDown support, and even comes with the possibility
to preview/render what you wrote. However, standard MarkDown and Julia
MarkDown do occasionally differ, so you will not be able to preview
everything. The VScode preview also only works on `.md`-files, so not on
docstrings.

## Adding the file to the docs.
While we have a file with MarkDown text, it does not yet show up in the
build docs. To make this happen, go to the `make.jl`-file in the
`docs/`-directory.
!!! warning "Do not change the name of the make.jl and index.jl files!"
    These files are needed by the `Documenter` package and need to have
    this filename. The build will fail if these are renamed (there are
    possibilities for doing this in more advanced setups, but we don't
    use these at the moment).
As you can see in the `make.jl`-file, the structure of the documentation
is defined here. The titles in the left menu are defined here. The title
of a specific page, however, is defined in the file for the page. Add
the new file that you want to add in the right place, and (re-)build the
docs to see the result (See the '[How to locally build the MANTIS docs.](@ref)'
page on how to build the docs).

## Adding docstring to the docs.
The general MarkDown files create the pages of the documentation. Of
course, the docs need to describe the code. To do this, it is possible
to include the docstrings into the documentation. An example of what
this looks like is the following:

```@meta
CurrentModule = Mantis.FunctionSpaces
```

```@docs; canonical=false
Bernstein
```

As you will see in the MarkDown file for this page, there are two blocks;
one `@meta`-block, where the module is specified, and one `@docs`-block.
In the `@docs`-block, you specify the module/type/method that you want
to describe. In this specific case, the `canonical=false` option is
specified. This allows this docstring to be included in the documentation
multiple times. Without this option, the docstring can only be included
once. If you try to include it multiple times anyway, the build will fail
and the error message will show that you tried to include a function
multiple times. The reason for only being allowed to include it once
without the `canonical=false` option is simply; you can cross-reference
to other docstrings and the only way to know which one is referenced is
by only having it once without the `canonical=false` option.

You can put the `@docs`-block (as well as `@meta`-blocks) wherever you
want. The blocks will be rendered at the place where you put them.

I have currently used `@autodocs` in the [Developer Documentation](@ref)
page to list all the currently documented functions to see what this
looks like.

!!! warning "Every docstring must be included at least once in the docs."
    The documentation is set up in such a way that an error will occur
    if you do not include a docstring in the docs from one of the modules
    listed in the `modules`-option in the `makedocs`-function in the
    `make.jl`-file. This is to ensure that the documentation includes
    all documented functions.

## Combining functions with the same name.
You can also combine the docstrings from functions with the same name but different input
by using:

```@meta
CurrentModule = Mantis.FunctionSpaces
```

```@docs; canonical=false
evaluate
```

This may be clearer for these cases, though it can only be collapsed as
one big block.
