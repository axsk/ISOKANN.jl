# Installation

Install Julia (>=v1.10) using https://github.com/JuliaLang/juliaup

```
curl -fsSL https://install.julialang.org | sh
```

After restarting your shell you should be able to start the Julia REPL via the command `julia`.

In the REPL you can add `ISOKANN.jl` to your project by entering the package mode (type `]`) and typing

```julia
pkg> add https://github.com/axsk/ISOKANN.jl
pkg> test ISOKANN
```

Note that this can take a while on the first run as Julia downloads and precompiles all dependencies.

We plan on installing openmm automatically with ISOKANN. Right now, if you want to use openmm with ISOKANN you will need to make it available to PyCall.jl.

## Development

If you want to make changes to ISOKANN you should clone it into a directory

`git clone git@github.com:axsk/ISOKANN.jl.git`

Then start Julia in that directory, activate it with `]activate .`,
instantiate the dependencies with `]instantiate`.

You should then be able to run the tests with `]test`
or start `using ISOKANN`.

We strongly recommend `using Revise` before ISOKANN, so that your changes will automatically load in your current session.