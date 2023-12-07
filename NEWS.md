Let us try to keep a rough log of the changes we did.

07.12.23
- isokann was working, we had beautiful muller brown plots
- i changed a lot afterwards to glue it to the dipeptide, should test if mb is still converging

22.2.23
- fix the ominous u0 bug where all trajectories always started in the same point
- added a chifix plot

21.2.23
- use the avogadro pdb with connections (important!)
- switch to ISOSim type
- added some pairwisedists implementation fighting zygote
- comment out SimpleChains since their package precompilation was broken
- pick better T, dt defaults depending on gamma

16.2.23
- Added `scatter_ramachandran()` to visualize the learned chi values over the dihedrals
- Switched `defaultmodel()` to use pairwise atom dists as first layer input
- Switched optimiser to use L2-regularization (`OptimizerChain(WeightDecay(), Adam())`)

15.2.23
- export trajectories for visualization (`extractdata(), exportdata(), savecoords()`)
- added Alanine dipeptide `PDB_ACEMD()`
- switched to use the type `MollySDE` to encapsulate all integration parameters