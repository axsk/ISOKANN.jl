using ISOKANN

using StatsBase

# step 1: basic functionality

struct GRNSimulation <: IsoSimulation
end

function simulate_grn(initial_condition; paramters=nothing)
    #doing ssa
    x = copy(initial_condition)
    for i in 1:100
        x += rand(size(initial_condition))
    end
    return x
end

function propagate(xs, k)
    d, n = size(xs)
    ys = zeros(d, k, n)
    for i in 1:n
        for j in 1:k
            ys[:, k, i] = simulate_grn(xs[:, i])
        end
    end
    return ys
end

function initial_data(n=100, k=2)
    xs = rand(8, n)
    ys = propagate(xs, k)
    return xs, ys
end

function basic_isokann_run()
    xs, ys = initial_data()
    data = SimulationData(xs, ys)
    iso = Iso(data)
    run!(iso, 100) # run until convergence
    plot_training(iso) # check convergence
end



# step 1.5
# visualization / interpretation
# how can we make sense of the isokann result in the gre setting

# step 2
# adaptive sampling
# requires
# GRESimulation <: ISOKANN.Simulation
# with methods `propagate(sim::GRESimulation, xs, ...)`, ndim(sim::GRESimulation)
# to work via kde uniform sampling

# step2.5 alternative to 2
# bruteforce lots of data (long trajectories)
# use kmeans / picking-algorithm to subsample starting positions
# construct data from starting positions and propagated positions

# step 3 # multidimensional (?)
# requires work by alex


# step 4
# put this together as isokann extension module/pkg
# ISOKANNGRN.jl

# step 5
# writing paper
