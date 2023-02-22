Stepsizes for the Dialanine system
- EM, gamma = 1: dt = 5e-6 seems to be the lowest working (0.24 sec for T=.01)
- SR, gamma = 1: dt = 5e-6 working, but needs 1.74 secs for same length
- SR, gamma = 1: dt = 2e-5 working, 500 ms (sde benchmarktime)

now with half stepsize for safety and benchmarking
- em gamma 1 dt 2e-6 570ms
- sr gamma 1 dt 1e-5 900ms

lets try gamma=10
- sr allows 10x bigger dt and needs about 10x longer to reach same "ramachandran spread"
- em kinda same


timing/cpu

30 ps simulation 1 thread = 16 secs