# Mantis

[![Build Status](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CI.yml?query=branch%3Amain)


Welcome to `MANTIS.jl`, a package for high-order structure-preserving 
finite element methods. This package is designed around 
structure-preserving/mimetic methods. It has extensive support for basis 
functions; 'standard' mimetic finite elements, multi-patch splines, and
([MIT license](https://github.com/MantisFEM/Mantis.jl/blob/main/LICENSE)).


## Contents
A brief description of what you can find in the `README.md`.
- A brief description of the Authors.
- A step-by-step guide on how to build and view the `Mantis` docs.
- An overview of the source code organisation.


## Authors
The `Mantis` package was created by
- Diogo Costa Cabanas,
- Joey Dekker,
- Artur Palha,
- Deepesh Toshniwal,

from TU Delft's Institute of Applied Mathematics (DIAM).

## Related publications.
One can hope.


## How to build the MANTIS docs.
As the documentation is not (yet) hosted online, you will have to build 
and view the documentation locally. How to do this is described here.

The main file structure of the the docs has now been created, so you 
only have to build the documentation. Fortunately, the heavy lifting 
will be done by `Julia` and `Python`.

To create the docs, follow these steps:
- Navigate to the `docs/` directory. If you open a terminal in the 
  `Mantis.jl` repo, this is a simple matter of executing 
  ```
  cd docs
  ```
  (or the equivalent on your system) in your terminal.
- Next, run 
  ```
  julia --color=yes --project make.jl
  ``` 
  in this directory (you can leave the `--color=yes` option out if you 
  don't want the printed output to be coloured). This will update (or 
  generate if it doesn't exist yet) a `build/`-directory in the 
  `docs/`-directory[^NoteOnGITtingBuildFolder]. All `.html` files are 
  generated here. *Make sure that you save all changes before running 
  this command or you won't see the changes!* Should this step fail, see 
  the next step.
- If the dependencies of `Mantis.jl` changed, the build may fail when 
  building in the `docs/`-folder. This is because the `docs/`-folder 
  defines its own julia environment, since the documentation may have 
  different dependencies than Mantis itself. To update the environment, 
  open a terminal in the `docs/`-directory and type `julia` to open 
  julia. Enter the package manager by typing `]` and activate the 
  current environment by typing `activate .`. The environment is shown 
  in parenthesis '()' and should say 'docs'. Type `dev ../../Mantis.jl` 
  and execute this command. The environment will be updated so that the 
  latest version of Mantis is available with its updated structure. Then 
  you may also have to run `instantiate` to make sure the 
  `docs\`-environment is updated. You can now redo the previous step.
- Then you need to create a (local) webserver to view the HTML docs, for 
  which there are a few options, see the [Documenter Docs](https://documenter.juliadocs.org/stable/man/guide/). 
  Since I have `Python` installed, I use the python option. Run
  ```
  python -m http.server --bind localhost
  ``` 
  (after, in my case, activating my conda environment by executing 
  `conda activate`) in the `docs/`-directory. 
- In my case, this will result in VScode giving me the message: `Your 
  application running on port 8000 is available. See all forwarded ports`
  with the options `Open in Browser` and `Preview in Editor`. Click the 
  `Open in Browser`-option to see the html pages. You have to click the 
  `build/`-link when the browser opens. 
- Enjoy the Mantis docs!
- When done, you can kill the local webserver by using `Ctrl+C`.

[^NoteOnGITtingBuildFolder]:
    The `docs/build/`-folder should **never** be gitted. (See also the 
    warning in the [Documenter Docs](https://documenter.juliadocs.org/stable/man/guide/)). 
    The `.gitignore`-file is already setup to prevent this.


## Source Code Organization

The Mantis source code is organized as follows:

| Directory         | Contents                                         |
| -                 | -                                                |
| `docs/`           | source for the user and developer manuals        |
| `src/`            | source for the all Mantis code                   |
| `test/`           | test suites                                      |
