Let us try to keep a rough log of the changes we did.

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