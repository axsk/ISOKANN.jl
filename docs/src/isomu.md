# IsoMu

Analysing trajectory data from mu opiod receptor with ISOKANN and reaction path subsampling.

contact: a sikorski, s chewle


# Loading the julia project

1. run julia (install via: google juliaup)
2. activate the project `julia> ]activate .`
3. update ISOKANN to their github master branches
- `julia> ]add https://github.com/JuliaMolSim/Molly.jl`
- `julia> ]add https://github.com/axsk/ISOKANN.jl`
4. load the module via `julia> using ISOKANN.IsoMu`

# Running the clustering
```julia
# create a DataLink to the trajectory's directory
link = DataLink("path/to/traj")

# create the ISOKANN environment
mu = isokann(link)

# train the network
train!(mu)

# save the reactive path
save_reactive_path(mu, out="out/path.pbd")

```


# Starting on SLURM with gpu
srun --gres=gpu --partition gpu --constraint "A40-RTX-48GB" --pty bash

then `ISOKANN.gpu!(mu::IsoRun)`

# A more advanced example
```julia
using IsoMu, Flux

# read the trajectory from the 10th frame, every 10 frames with distance cutoff 10 and reverse the trajectory
data = DataLink("data/8EF5_500ns_pka_7.4_no_capping_310.10C/", startpos=10, stride=10, radius=10, reverse=true)

# specify the network and training parameters
mu = isokann(data, networkargs=(;layers=4, activation=Flux.leakyrelu), learnrate = 1e-3, regularization=1e-4, minibatch=256,)

gpu!(mu)  # transfer model to gpu
train!(mu, 10000)  # 10000 iterations
adjust!(mu, 1e-4, lambda=1e-3) # set learnrate to 1e-4 and decay to 1e-3
train!(mu, 10000)  # 10000 iterations
```
