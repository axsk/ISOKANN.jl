New features:
- support for particle-weighted alignment/rmsd
- `data_from_trajectory` now supports an additional `lag` parameter (on top of `stride`)
- `dchidfeat` returns derivative of the model wrt. the features
- `mutual_information(iso)` computes the mutual information between chi and features


Improvements and fixes:
- Make the `WGLMakie` dependency optional
- refactor utility code to `utils`
- simplified handling of external `SimulationData` through implicit use of `ExternalSimulation`
- more robust dispatch for `exit_rates`
- `ExternalSimulation` now stores parameters in a `Dict`
- `FeaturesPairs` now supports `mdtraj` selection syntax
- `propagate` has inbuild retry mechanic
- `reactive_path` now allows to distinguish between `coords` for the distance computations and `fullcoords` for the output, as well as supporting weights
- `shortestchain` now masks the distance computations, improving `reactive_path` compute time for tight `minjump`/`maxjump` bounds