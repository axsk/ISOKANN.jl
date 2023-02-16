Let us try to keep a rough log of the changes we did.

16.2.23
- Added `scatter_ramachandran()` to visualize the learned chi values over the dihedrals
- Switched `defaultmodel()` to use pairwise atom dists as first layer input
- Switched optimiser to use L2-regularization (`OptimizerChain(WeightDecay(), Adam())`)

15.2.23
- export trajectories for visualization (`extractdata(), exportdata(), savecoords()`)
- added Alanine dipeptide `PDB_ACEMD()`
- switched to use the type `MollySDE` to encapsulate all integration parameters