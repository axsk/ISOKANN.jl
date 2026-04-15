# Installation

## Julia

Install Julia (v1.11 or newer) with [juliaup](https://github.com/JuliaLang/juliaup):

```
curl -fsSL https://install.julialang.org | sh
```

After restarting your shell, `julia` starts the REPL.

## ISOKANN

In the REPL, enter package mode (type `]`) and add the release version:

```julia
pkg> add ISOKANN
pkg> test ISOKANN
```

For the latest development version:

```julia
pkg> add https://github.com/axsk/ISOKANN.jl
```

If you are new to Julia, take a moment to read about
[Pkg environments](https://pkgdocs.julialang.org/v1/environments/); ISOKANN is
most convenient when used inside a project environment.

The first run precompiles a sizeable dependency tree — expect a few minutes.

## OpenMM (for molecular dynamics)

Since v2.0, ISOKANN calls OpenMM through
[PythonCall.jl](https://juliapy.github.io/PythonCall.jl/), and Python
dependencies are managed by
[CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl). Installing ISOKANN should
bring OpenMM, `joblib`, and `openmmforcefields` in automatically — no manual
PyCall/Conda build step is needed.

If you want the same conda environment available from a Python shell too
(e.g. to use the Python bindings in [`python/`](https://github.com/axsk/ISOKANN.jl/tree/main/python)),
point your tooling at the environment CondaPkg provisions (typically under
`.CondaPkg/env/`), or share a single conda environment via CondaPkg's
`CONDAPKG_BACKEND` / shared-env settings.

For troubleshooting, see the
[CondaPkg](https://github.com/JuliaPy/CondaPkg.jl) and
[PythonCall](https://juliapy.github.io/PythonCall.jl/) docs, or open an issue on
the ISOKANN repository.

## Development

To hack on ISOKANN itself:

```
git clone git@github.com:axsk/ISOKANN.jl.git
cd ISOKANN.jl
julia --project=.
```

Then inside Julia:

```julia
pkg> instantiate
```

We strongly recommend loading `Revise` before `ISOKANN` so edits take effect
without restarting:

```julia
using Revise
using ISOKANN
```

See the [Julia Pkg docs](https://pkgdocs.julialang.org/v1/) for a deeper tour
of environments and development workflows.

## Editor

VS Code with the Julia extension works well. On a remote machine, the
Remote-SSH / Remote Coder extensions let you drive the same setup locally. The
package is designed around interactive use in the REPL, so keep one open
alongside your editor.

## GPU

Training and most analysis routines run on CUDA GPUs via Flux/Functors. Move
an `Iso` to and from GPU with `gpu(iso)` / `cpu(iso)`. The OpenMM simulation
itself selects its own platform (CPU/CUDA/OpenCL) via OpenMM's platform
mechanism.
