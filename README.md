# ISOKANN

This should be the reference implementation for ISOKANN.
Currently things are still fluctuating, so we have different implementations

- isokann.jl - full fledged ISOKANN with adaptive sampling and optimal control for overdamped Langevin systems
- isomolly.jl - basic ISOKANN with adaptive sampling for Molly.jl systems (e.g. proteins)

After installing the package with `Pkg.add("https://github.com/axsk/ISOKANN.jl")` run a basic alanine dipeptide run with

```julia

using ISOKANN

iso = IsoRun()  # create the ISOKANN system/configuration
run!(iso)       # run it for iso.nd steps

plot_learning(iso) # plot the training overview

# scatter plot of all initial points colored in  corresponding chi value
scatter_ramachandran(iso.data[1], iso.model) 

# estimate the eigenvalue, i.e. the metastability
eigenvalue(iso.model, iso.data[1])

```


The big todos:
1. Merge optimal control into isomolly (requires a new Molly.jl sampler)
2. Use PCCA+ for higher dimensional chi functions
3. Gather experience on larger molecules.

The most recent plot:
<img src="out/lastplot.png"/>
