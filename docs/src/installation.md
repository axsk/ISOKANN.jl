# Installation

Install Julia (v1.11 or newer) using https://github.com/JuliaLang/juliaup

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

ISOKANN is using OpenMM for sampling, which is being called through PyCall.jl - a library facilitating python calls from withing julia.
PyCall can install OpenMM automatically if it is using a conda environment managed by Julia through Conda.jl.
This is the default on Windows on Mac systems, on Linux however you would need to install OpenMM via pip yourself or instruct Julia to use the Conda.jl environment by running


Then instruct PyCall to use the Conda.jl environment by calling
```julia
julia> ENV["PYTHON"] = ""
pkg> build PyCall
```

To use a shared Conda environment, e.g. because of quota limitations, you can configure it using prior to building PyCall:
```julia
julia> ENV["CONDA_JL_HOME"] = "/data/[...]/conda/envs/conda_jl"  # change this to your path, which contains bin/conda etc.
pkg> build Conda
```

See also the [PyCall docs](https://github.com/JuliaPy/PyCall.jl?tab=readme-ov-file#specifying-the-python-version).

## Development

If you want to make changes to ISOKANN you should clone it into a directory

`git clone git@github.com:axsk/ISOKANN.jl.git`

Then start Julia in that directory, activate it with `]activate .`,
instantiate the dependencies with `]instantiate`.

See also the (almost mandatory read) on the [Julia Pkg docs](https://pkgdocs.julialang.org/v1/).

We strongly recommend using the Revise.jl package through `using Revise` before loading ISOKANN, so that your changes will automatically load in your current session.
