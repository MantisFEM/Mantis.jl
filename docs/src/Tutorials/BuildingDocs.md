# How to locally build the MANTIS docs.
The main file structure of the the docs has now been created, so you 
only have to build the documentation. Fortunately, the heavy lifting 
will be done by `Julia` and/or `Python`.

For convenience, there is a bash script in the tools folder, 
`build_local_docs.sh`, that automates the process. Make sure to run the
script starting in the main Mantis folder. Use
```
bash tools/build_local_docs.sh
```
If this does not work, or if you prefer to go through the steps yourself, 
you can use the following step-by-step guide.

To create the docs, follow these steps:
- Navigate to the `docs/` directory. If you open a terminal in the 
  `Mantis` repo, this is a simple matter of executing 
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
- If the dependencies of `Mantis` changed, the build may fail when 
  building in the `docs/`-folder. This is because the `docs/`-folder 
  defines its own julia environment, since the documentation may have 
  different dependencies than Mantis itself. To update the environment, 
  open a terminal in the `docs/`-directory and type `julia` to open 
  julia. Enter the package manager by typing `]` and activate the 
  current environment by typing `activate .`. The environment is shown 
  in parenthesis '()' and should say 'docs'. Type `dev ../../Mantis.jl` 
  and execute this command. **Note that `../../Mantis.jl` refers to the 
  folder name. If you did not call this Mantis.jl, make sure to use to 
  correct name.** The environment will be updated so that the latest 
  version of Mantis is available with its updated structure. Then you 
  may also have to run `instantiate` to make sure the `docs\`-environment 
  is updated. You can now redo the previous step.
- Then you need to create a (local) webserver to view the HTML docs, for 
  which there are a few options, see the [Documenter Docs](https://documenter.juliadocs.org/stable/man/guide/). 
  Since I have `Python` installed, I use the python option. Run
  ```
  python -m http.server --bind localhost
  ``` 
  (after, in my case, activating my conda environment by executing 
  `conda activate`) in the `docs/`-directory. You may have to use
  ```
  python3 -m http.server --bind localhost
  ``` 
  instead. Alternatively, you can use Julia with `LiveServer.jl`:
  ```
  julia -e 'using LiveServer; serve(dir="build")
  ``` 
  Note that the automate script tries all these options for you.
- In my case, this will result in VScode giving me the message: `Your 
  application running on port 8000 is available. See all forwarded ports`
  with the options `Open in Browser` and `Preview in Editor`. Click the 
  `Open in Browser`-option to see the html pages. You (may) have to click 
  the `build/`-link when the browser opens. 
- Enjoy the Mantis docs! 
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
