# Installation

Install Julia (>=v1.10) using https://github.com/JuliaLang/juliaup

```
curl -fsSL https://install.julialang.org | sh
```

After restarting your shell you should be able to start the Julia REPL via the command `julia`.

In the REPL you can add (the release version of) `ISOKANN.jl` to your project by entering the package mode (type `]`) and typing

```julia
pkg> add ISOKANN
pkg> test ISOKANN
```

If you want the newest features you can instead install the newest version using `pkg> add https://github.com/axsk/ISOKANN.jl` instead.

Note that this can take a while on the first run as Julia downloads and precompiles all dependencies.

We plan on installing OpenMM automatically with ISOKANN. Right now, if you want to use openmm with ISOKANN you will need to make it available to PyCall.jl.
This should work automatically, when Julia is using its own Conda.jl Conda environment. In order to set this up (once) run: 
- `shell> PYTHON="" julia` to start julia with the environment variable indicating to use its own Python environment
- `pkg> build PyCall.` to rebuild PyCall

See also the [PyCall docs](https://github.com/JuliaPy/PyCall.jl?tab=readme-ov-file#specifying-the-python-version).

## Development

If you want to make changes to ISOKANN you should clone it into a directory

`git clone git@github.com:axsk/ISOKANN.jl.git`

Then start Julia in that directory, activate it with `]activate .`,
instantiate the dependencies with `]instantiate`.

See also the (almost mandatory read) on the [Julia Pkg docs](https://pkgdocs.julialang.org/v1/).

We strongly recommend using the Revise.jl package through `using Revise` before loading ISOKANN, so that your changes will automatically load in your current session.