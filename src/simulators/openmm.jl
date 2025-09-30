module OpenMM

using PyCall, CUDA
using LinearAlgebra: norm, dot, diag
using Random: randn!

import JLD2
import ProgressMeter
import ..ISOKANN: ISOKANN, IsoSimulation,
    propagate, dim, randx0,
    featurizer, defaultmodel,
    savecoords, coords, force, pdbfile,
    force, potential, lagtime, trajectory, laggedtrajectory,
    WeightedSamples, masses

export OpenMMSimulation, FORCE_AMBER, FORCE_AMBER_IMPLICIT
export OpenMMScript
export FeaturesAll, FeaturesAll, FeaturesPairs, FeaturesRandomPairs

export trajectory, propagate, setcoords, coords, savecoords
export atoms

DEFAULT_PDB = normpath("$(@__DIR__)/../../data/systems/alanine dipeptide.pdb")
FORCE_AMBER = ["amber14-all.xml"]
FORCE_AMBER_IMPLICIT = ["amber14-all.xml", "implicit/obc2.xml"]
FORCE_AMBER_EXPLICIT = ["amber14-all.xml", "amber/tip3p_standard.xml"]

global OPENMM

function __init__()
    # install / load OpenMM
    try
        OPENMM = pyimport_conda("openmm", "openmm", "conda-forge")
        pyimport_conda("openmmforcefields", "openmmforcefields", "conda-forge")
        pyimport_conda("joblib", "joblib")

        @pyinclude("$(@__DIR__)/mopenmm.py")
    catch
        @warn "Could not load openmm."
    end
end

###

#abstract type OpenMMSimulation <: IsoSimulation end

"""
    OpenMMSimulation(; pdb, steps, ...)
    OpenMMSimulation(; py, steps)

Constructs an OpenMM simulation object.
Either use  `OpenMMSimulation(;py, steps)` where `py`` is the location of a .py python script creating a OpenMM simulation object
or supply a .pdb file via `pdb` and the following parameters (see also defaultsystem):

## Arguments
- `pdb::String`: Path to the PDB file.
- `ligand::String`: Path to ligand file.
- `forcefields::Vector{String}`: List of force field XML files.
- `temp::Float64`: Temperature in Kelvin.
- `friction::Float64`: Friction coefficient in 1/picosecond.
- `step::Float64`: Integration step size in picoseconds.
- `steps::Int`: Number of simulation steps.
- `features`: Which features to use for learning the chi function.
              -  A vector of `Int` denotes the indices of all atoms to compute the pairwise distances from.
              -  A vector of CartesianIndex{2} computes the specific distances between the atom pairs.
              -  A number denotes the radius below which all pairs of atoms will be used (computed only on the starting configuration)
              -  If `nothing` all pairwise distances are used.
- `minimize::Bool`: Whether to perform energy minimization on first state.
- `nthreads`: The number of threads to use for parallelization of multiple simulations.
- `mmthreads`: The number of threads to use for each OpenMM simulation. Set to "gpu" to use the GPU platform.

## Returns
- `OpenMMSimulation`: An OpenMMSimulation object.

"""
struct OpenMMSimulation{T} <: IsoSimulation
    pysim::PyObject
    steps::Int
    constructor
    bias::T
end

function OpenMMSimulation(; steps=100, bias::T=nothing, k...) where {T}
    k = NamedTuple(k)
    if haskey(k, :py)
        py"""
        julia=True
        """
        @pyinclude(k.py)
        pysim = py"simulation"

        return OpenMMSimulation{T}(pysim, steps, k, bias)
    elseif haskey(k, :pdb)
        pysim = defaultsystem(; k...)
        constructor = (; k..., steps, bias)
        return OpenMMSimulation{T}(pysim, steps, constructor, bias)
    else
        return OpenMMSimulation(; pdb=DEFAULT_PDB, bias, steps, k...)
    end
end


# NOTE: make sure to use forcefield_kwargs=Dict(:flexibleConstraints=>true) when computing the reactive path with explicit water
# c.f. https://github.com/openmm/openmm/issues/4670
function defaultsystem(;
    pdb=DEFAULT_PDB,
    ligand="",
    forcefields=["amber14-all.xml"],
    temp=310, # kelvin
    friction=1,  # 1/picosecond
    step=0.002, # picoseconds
    addwater=false,
    minimize=addwater,
    padding=1,
    ionicstrength=0.0,
    mmthreads=CUDA.functional() ? "gpu" : 1,
    forcefield_kwargs=Dict(),
    kwargs...
)
    @pycall py"defaultsystem"(pdb, ligand, forcefields, temp, friction, step, minimize; addwater, padding, ionicstrength, forcefield_kwargs, mmthreads, kwargs...)::PyObject
end


# TODO: this are remnants of the old detailed openmm system, remove them eventually
mmthreads(sim::OpenMMSimulation) = get(sim.constructor, :mmthreads, CUDA.functional() ? "gpu" : 1)
nthreads(sim::OpenMMSimulation) = get(sim.constructor, :nthreads, CUDA.functional() ? 1 : Threads.nthreads())
pdbfile(sim::OpenMMSimulation) = get(() -> createpdb(sim), sim.constructor, :pdb)

steps(sim) = sim.steps::Int
friction(sim) = friction(sim.pysim)
temp(sim) = temp(sim.pysim)
stepsize(sim) = stepsize(sim.pysim)


lagtime(sim::OpenMMSimulation) = steps(sim) * stepsize(sim) # in ps
dim(sim::OpenMMSimulation) = length(coords(sim))
defaultmodel(sim::OpenMMSimulation; kwargs...) = ISOKANN.pairnet(; kwargs...)

coords(sim::OpenMMSimulation) = coords(sim.pysim)::Vector{Float64}
setcoords(sim::OpenMMSimulation, coords) = setcoords(sim.pysim, coords)
natoms(sim::OpenMMSimulation) = div(dim(sim), 3)

friction(pysim::PyObject) = pysim.integrator.getFriction()._value # 1/ps
temp(pysim::PyObject) = pysim.integrator.getTemperature()._value # kelvin
stepsize(pysim::PyObject) = pysim.integrator.getStepSize()._value # ps
coords(pysim::PyObject) = pysim.context.getState(getPositions=true, enforcePeriodicBox=true).getPositions(asNumpy=true).flatten()

iscuda(sim::OpenMMSimulation) = iscuda(sim.pysim)
iscuda(pysim::PyObject) = pysim.context.getPlatform().getName() == "CUDA"
claim_memory(sim::OpenMMSimulation) = iscuda(sim) && CUDA.functional() && CUDA.reclaim()

function createpdb(sim)
    pysim = sim.pysim
    file = tempname() * ".pdb"
    pdb = py"app.PDBFile"
    pdb.writeFile(pysim.topology, PyReverseDims(reshape(coords(sim), 3, :)), file)  # TODO: fix this
    return file
end

featurizer(sim::OpenMMSimulation) = featurizer(sim, get(sim.constructor, :features, nothing))

featurizer(sim, ::Nothing) =
    if natoms(sim) < 100
        FeaturesAll()
    else
        maxfeatures = 100
        @warn("No default featurizer specified. Falling back to $maxfeatures random pairs")
        FeaturesPairs(sim; maxdist=0, maxfeatures)
    end
featurizer(sim, atoms::Vector{Int}) = FeaturesAtoms(atoms)
featurizer(sim, pairs::Vector{Tuple{Int,Int}}) = FeaturesPairs(pairs)
featurizer(sim, features::Function) = features
featurizer(sim, radius::Number) = FeaturesPairs([calpha_pairs(sim.pysim); local_atom_pairs(sim.pysim, radius)] |> unique)

struct FeaturesCoords end
(f::FeaturesCoords)(coords) = coords

struct FeaturesAll end
(f::FeaturesAll)(coords) = ISOKANN.flatpairdists(coords)

""" Pairwise distances between all provided atoms """
struct FeaturesAtoms
    atominds::Vector{Int}
end
(f::FeaturesAtoms)(coords) = ISOKANN.flatpairdists(coords, f.atominds)

struct FeaturesPairs
    pairs::Vector{Tuple{Int,Int}}
end
(f::FeaturesPairs)(coords) = ISOKANN.pdists(coords, f.pairs)
Base.show(io::IO, f::FeaturesPairs) = print(io, "FeaturesPairs() with $(length(f.pairs)) features")


import StatsBase

"""
    FeaturesPairs(pairs::Vector{Tuple{Int,Int}})
    FeaturesPairs(system; selector="all", maxdist=Inf, maxfeatures=Inf)

Creates a FeaturesPairs object from either:
- a list of index pairs (`Vector{Tuple{Int,Int}}`) passed directly.
- an `OpenMMSimulation` or PDB file path (`String`), selecting atom pairs using MDTraj selector syntax (`selector`),
  optionally filtered by `maxdist` (in nm) and limited to `maxfeatures` randomly sampled pairs.
"""
FeaturesPairs(sim::OpenMMSimulation; kwargs...) = FeaturesPairs(pdbfile(sim), kwargs...)
function FeaturesPairs(pdb::String; selector="all", maxdist=Inf, maxfeatures=Inf)
    mdtraj = pyimport_conda("mdtraj", "mdtraj", "conda-forge")
    m = mdtraj.load(pdb)
    inds = m.top.select(selector) .+ 1
    if maxdist < Inf
        c = permutedims(m.xyz, (3, 2, 1))
        c = reshape(c, :, size(c, 3))
        inds = ISOKANN.restricted_localpdistinds(c, maxdist, inds)
    else
        inds = [(inds[i], inds[j]) for i in 1:length(inds) for j in i+1:length(inds)]
    end
    if length(inds) > maxfeatures
        inds = StatsBase.sample(inds, maxfeatures, replace=false) |> sort
    end
    return FeaturesPairs(inds)
end

import BioStructures
struct FeaturesAngles
    struc
end

function FeaturesAngles(sim::OpenMMSimulation)
    return FeaturesAngles(read(sim.constructor.pdb, BioStructures.PDBFormat))
end

function (f::FeaturesAngles)(coords::AbstractVector)
    coords = reshape(coords, 3, :)
    atoms = collectatoms(f.struc)
    for (a, c) in zip(atoms, eachcol(coords))
        coords!(a, c)
    end
    filter(!isnan, vcat(phiangles(f.struc), psiangles(f.struc)))
end

function (f::FeaturesAngles)(coords)
    mapslices(f, coords, dims=1)
end




""" generate `n` random inintial points for the simulation `mm` """
randx0(sim::OpenMMSimulation, n) = ISOKANN.laggedtrajectory(sim, n)

"""
    propagate(sim::OpenMMSimulation, x0::AbstractMatrix, nk)

Propagates `nk` replicas of the OpenMMSimulation `sim` from the inintial states `x0`.

# Arguments
- `sim`: An OpenMMSimulation object.
- `x0`: Matrix containing the initial states as columns
- `nk`: The number of replicas to create.

"""
function propagate(sim::OpenMMSimulation, x0::AbstractMatrix, nk; retries=3)
    claim_memory(sim)
    dim, nx = size(x0)
    ys = isnothing(sim.bias) ? similar(x0, dim, nk, nx) : WeightedSamples(similar(x0, dim, nk, nx), zeros(1, nk, nx))
    p = ProgressMeter.Progress(nk * nx, desc="Propagating")
    for i in 1:nx
        for j in 1:nk
            with_retries(retries, PyCall.PyError) do 
                    ys[:, j, i] = laggedtrajectory(sim, 1, x0=x0[:, i], throw=true, showprogress=true, reclaim=false)
            end
            ProgressMeter.next!(p)
        end
    end
    return ys
end

function with_retries(f, retries, allowed_error=Any)
    for trial in 1:retries
        try
            return f()
        catch e 
            if e isa allowed_error && trial < retries
                @warn "retrying on error $e"
                lasterr = e
                continue
            else
                rethrow(e)
            end
        end
    end
end

"""
    laggedtrajectory(sim::OpenMMSimulation, lags; steps=steps(sim), resample_velocities=true, kwargs...)

Generate a lagged trajectory for a given OpenMMSimulation.
E.g. x0--x--x--x  for `lags=3` and `steps=2`

# Arguments
- `sim::OpenMMSimulation`: The simulation object.
- `lags`: The number of steps.
- `steps`: The lagtime, i.e. number of steps to take in the simulation.
- `resample_velocities`: Whether to resample velocities according to Maxwell-Boltzman for each lag.
- `kwargs...`: Additional keyword arguments to pass to the `trajectory` function.

# Returns
- A matrix of `lags` samples which each have `steps` simulation-steps inbetween them.
"""
laggedtrajectory(sim::OpenMMSimulation, lags; steps=steps(sim), resample_velocities=true, kwargs...) =
    trajectory(sim, lags * steps; saveevery=steps, resample_velocities, kwargs...)


"""
    trajectory(sim::OpenMMSimulation{Nothing}, steps=steps(sim); saveevery=1, x0=coords(sim), resample_velocities=false, throw=false, showprogress=true, reclaim=true)

Simulates the trajectory of an OpenMM simulation.

# Arguments
- `sim::OpenMMSimulation{Nothing}`: The OpenMM simulation object.
- `steps`: The number of steps to simulate. Defaults to the number of steps defined in the simulation object.
- `saveevery`: Interval at which to save the trajectory. Defaults to 1.
- `x0`: Initial coordinates for the simulation. Defaults to the current coordinates of the simulation object.
- `sample_velocities`: Whether to sample velocities at the start of the simulation.
- `resample_velocities`: Whether to resample velocities after each `saveevery` steps. Defaults to `false`.
- `throw`: Whether to throw an error if the simulation fails. If false it returns the trajectory computed so far. Defaults to `false`.
- `showprogress`: Whether to display a progress bar during the simulation. Defaults to `true`.
- `reclaim`: Whether to reclaim CUDA memory before the simulation. Defaults to `true`.

# Returns
- The trajectory of the simulation as a matrix of coordinates.
"""
function trajectory(sim::OpenMMSimulation{Nothing}, steps=steps(sim); saveevery=1, x0=coords(sim), sample_velocities=true, resample_velocities=false, throw=false, showprogress=true, reclaim=true)
    reclaim && claim_memory(sim)
    n = div(steps, saveevery)
    xs = similar(x0, length(x0), n)
    int = sim.pysim.context.getIntegrator()

    p = ProgressMeter.Progress(n, "Computing trajectory")
    done = 0
    runtime = 0.0
    lagtime = stepsize(sim) * saveevery / 1000
    tottime = stepsize(sim) * steps / 1000

    setcoords(sim, x0)
    sample_velocities && set_random_velocities!(sim)

    try
        for i in 1:n
            resample_velocities && set_random_velocities!(sim)
            runtime += @elapsed sim.pysim.step(saveevery)
            #runtime += @elapsed int.step(saveevery)  # this does not trigger loggers
            xs[:, i] = coords(sim)
           # @assert norm(xs[:, i]) <= 1e5
            done = i

            simtime = round(lagtime * i, sigdigits=3)
            showprogress && ProgressMeter.next!(p; showvalues=[("simulated time", "$simtime / $tottime ns"),
                ("speed", "$(simtime/runtime) ns/s"), ("norm", norm(xs[:, i]))])
        end
    catch e
        throw && rethrow()
        println()
        st = (e isa InterruptException) ? "InterruptException() " : sprint(showerror, e, catch_backtrace()) * "\n"
        @warn """$(st)during trajectory simulation.
        Returing partial result consisting of $done samples """
        return xs[:, 1:done]
    end
    return xs
end

function minimize!(sim::OpenMMSimulation, coords=coords(sim); iter=0)
    setcoords(sim, coords)
    return sim.pysim.minimizeEnergy(maxIterations=iter)
    return nothing
end

function setcoords(sim::PyObject, coords::AbstractVector{T}) where {T}
    c = PyReverseDims(reshape(coords, 3, :))
    sim.context.setPositions(c)::Nothing
    return nothing
end

function set_random_velocities!(sim::OpenMMSimulation)
    context = sim.pysim.context
    context.setVelocitiesToTemperature(context.getIntegrator().getTemperature())
end

force(sim::OpenMMSimulation, x::CuArray; kwargs...) = force(sim, Array(x); kwargs...) |> cu
potential(sim::OpenMMSimulation, x::CuArray; kwargs...) = potential(sim, Array(x); kwargs...)
masses(sim::OpenMMSimulation) = [sim.pysim.system.getParticleMass(i - 1)._value for i in 1:sim.pysim.system.getNumParticles()] # in daltons

function force(sim::OpenMMSimulation, x; reclaim=true)
    reclaim && claim_memory(sim)
    setcoords(sim, x)
    pysim = sim.pysim
    pyarray = PyArray(py"$pysim.context.getState(getForces=True).getForces(asNumpy=True)._value.reshape(-1)"o)
    return pyarray
end

potential(sim::OpenMMSimulation, x; kwargs...) = mapslices(x, dims=1) do x potential(sim, x; kwargs...) end

function potential(sim::OpenMMSimulation, x::AbstractVector; reclaim=true)
    reclaim && claim_memory(sim)
    setcoords(sim, x)
    v = sim.pysim.context.getState(getEnergy=true).getPotentialEnergy()
    v = v.value_in_unit(v.unit)
end


"""
    savecoords(path, sim::OpenMMSimulation, coords::AbstractArray{T})

Save the given `coordinates` in a .pdb file using OpenMM
"""
function savecoords(path, sim::OpenMMSimulation, coords::AbstractArray{T}=coords(sim)) where {T}
    coords = ISOKANN.cpu(coords)
    s = sim.pysim
    p = py"pdbfile.PDBFile"
    file = py"open"(path, "w")
    p.writeHeader(s.topology, file)
    for (i, coords) in enumerate(eachcol(coords))
        pos = reinterpret(Tuple{T,T,T}, coords .* 10)
        p.writeModel(s.topology, pos, file, modelIndex=i)
    end
    p.writeFooter(s.topology, file)
    file.close()
end

### FEATURE FILTERS

function remove_H_H2O_NACL(atom)
    !(
        atom.element.symbol == "H" ||
        atom.residue.name in ["HOH", "NA", "CL"]
    )
end

atoms(sim::OpenMMSimulation) = collect(sim.pysim.topology.atoms())

function local_atom_pairs(pysim::PyObject, radius; atomfilter=remove_H_H2O_NACL)
    xs = reshape(coords(pysim), 3, :)
    atoms = filter(atomfilter, pysim.topology.atoms() |> collect)
    inds = map(atom -> atom.index + 1, atoms)

    pairs = Tuple{Int,Int}[]
    for i in 1:length(inds)
        for j in i+1:length(inds)
            if norm(xs[:, i] - xs[:, j]) <= radius
                push!(pairs, (inds[i], inds[j]))
            end
        end
    end
    return pairs
end

function calpha_pairs(pysim::PyObject)
    local_atom_pairs(pysim, Inf; atomfilter=x -> x.name == "CA")
end

calpha_inds(sim::OpenMMSimulation) = calpha_inds(sim.pysim)
function calpha_inds(pysim::PyObject)
    map(filter(x -> x.name == "CA", pysim.topology.atoms() |> collect)) do atom
        atom.index + 1
    end
end


### SERIALIZATION

struct OpenMMSimulationSerialized
    constructor
end

JLD2.writeas(::Type{T}) where {T<:OpenMMSimulation} = OpenMMSimulationSerialized

Base.convert(::Type{OpenMMSimulationSerialized}, sim::OpenMMSimulation) =
    OpenMMSimulationSerialized(sim.constructor)

Base.convert(::Type{OpenMMSimulation{T}}, s::OpenMMSimulationSerialized) where {T<:Any} =
try
    OpenMMSimulation(; s.constructor...)
catch
    @warn "Could not reconstruct OpenMMSimulation(; $(s.constructor))"
    OpenMMSimulation()
end

Base.show(io::IO, mime::MIME"text/plain", sim::OpenMMSimulation) =
    print(io, "OpenMMSimulation(; $(string(sim.constructor)[2:end-1]))")


### CUSTOM INTEGRATORS

"""
    integrate_langevin(sim::OpenMMSimulation, x0=coords(sim); steps=steps(sim), bias::Union{Function,Nothing}=nothing, saveevery::Union{Int, nothing}=nothing)

Integrate the Langevin equations with a Euler-Maruyama scheme, allowing for external forces.

- bias: An additional force perturbation. It is expected to have the form bias(F, x) and mutating the provided force F.
- saveevery: If `nothing`, returns just the last point, otherwise returns an array saving every `saveevery` frame.
"""
function integrate_langevin(sim::OpenMMSimulation, x0=coords(sim); steps=steps(sim), bias::Union{Function,Nothing}=nothing, saveevery::Union{Int,Nothing}=nothing, reclaim=true)
    reclaim && claim_memory(sim)
    # we use the default openmm units, i.e. nm, ps
    x = copy(x0)
    v = zero(x) # this should be either provided or drawn from the Maxwell Boltzmann distribution
    kBT = 0.008314463 * temp(sim)
    dt = stepsize(sim)
    gamma = friction(sim)
    m = repeat(masses(sim), inner=3)
    out = isnothing(saveevery) ? x : similar(x, length(x), cld(steps, saveevery))

    for i in 1:steps
        F = force(sim, x, reclaim=false)
        isnothing(bias) || (bias(F, x)) # note that bias(F, x) is assumed to be mutating F
        langevin_step!(x, v, F, m, gamma, kBT, dt)
        isnothing(saveevery) || i % saveevery == 0 && (out[:, div(i, saveevery)] = x)
    end
    return out
end

function langevin_step!(x, v, F, m, gamma, kBT, dt)
    db = randn(length(x))
    @. v += 1 / m * ((F - gamma * v) * dt + sqrt(2 * gamma * kBT * dt) * db)
    @. x += v * dt
end

function integrate_girsanov(sim::OpenMMSimulation; x0=coords(sim), steps=steps(sim), bias, reclaim=true)
    reclaim && claim_memory(sim)
    # TODO: check units on the following three lines
    kB = 0.008314463
    dt = stepsize(sim)
    γ = friction(sim)

    M = repeat(masses(sim), inner=3)
    T = temp(sim)
    σ = @. sqrt(2 * kB * T / (γ * M))

    x = copy(x0)
    g = 0.0

    z = similar(x, length(x), steps)

    for i in 1:steps
        F = force(sim, x, reclaim=false)
        ux = bias(x)
        g += od_langevin_step_girsanov!(x, F, M, σ, γ, dt, ux)
        z[:, i] = x
    end

    return x, g, z
end

function od_langevin_step_girsanov!(x, F, M, σ, γ, dt, u)
    dB = randn(length(x)) * sqrt(dt)
    @. x += (1 / (γ * M) * F + (σ * u)) * dt + σ * dB
    dg = dot(u, u) / 2 * dt + dot(u, dB)
    return dg
end


trajectory(sim::OpenMMSimulation, steps=steps(sim); kwargs...) = langevin_girsanov!(sim, steps; kwargs...)

function langevin_girsanov!(sim::OpenMMSimulation, steps=steps(sim); bias=sim.bias, saveevery=1, x0=coords(sim), resample_velocities=false, showprogress=true, throw=true, reclaim=true)
    reclaim && claim_memory(sim)
    prog = ProgressMeter.Progress(steps)
    nout = div(steps, saveevery)
    qs = similar(x0, length(x0), nout)
    gs = zeros(1, nout)

    # ABOBA scheme from from https://pubs.acs.org/doi/full/10.1021/acs.jpcb.4c01702
    kB = 0.008314463
    dt = stepsize(sim)
    ξ = friction(sim)
    T = temp(sim)
    M = repeat(masses(sim), inner=3)

    # Maxwell-Boltzmann distribution
    p = randn(length(M)) .* sqrt.(M .* kB .* T)

    σ = @. sqrt(2 * kB * T * ξ * M)

    t2 = dt / 2
    a = @. t2 / M # eq 18
    d = @. exp(-ξ * dt) # eq 17
    f = @. sqrt(kB * T * M * (1 - exp(-2 * ξ * dt))) # eq 17
    q = x0

    b = similar(p)
    η = similar(p)
    Δη = similar(p)
    g = 0

    for k in 1:steps
        randn!(η)
        @. q += a * p # A
        F = force(sim, q, reclaim=false)
        B = bias(q; t=(k - 1) * dt, sigma=σ, F=F) # perturbation force ∇U_bias = -F
        @. Δη = (d + 1) / f * dt / 2 * B
        g += η' * Δη + Δη' * Δη / 2
        F .+= B  # total force: -∇V - ∇U_bias

        @. b = t2 * F
        @. p += b  # B
        @. p = d * p + f .* η # O
        @. p += b  # B
        @. q += a * p # A

        if k % saveevery == 0
            let i = div(k, saveevery)
                qs[:, i] = q
                gs[1, i] = g
            end
            resample_velocities && (p = randn(length(M)) .* sqrt.(M .* kB .* T))  # note that here the girsanov weights become meaningless
        end

        showprogress && ProgressMeter.next!(prog)
    end
    return WeightedSamples(qs, exp.(-gs))
end

# optimal control for sampling of chi function with OVERDAMPED langevin
function optcontrol(iso, forcescale=1.0)
    sim = iso.data.sim

    kB = 0.008314463
    γ = friction(sim)
    M = repeat(masses(sim), inner=3)
    TT = temp(sim)
    σ = @. sqrt(2 * kB * TT / (γ * M)) # ODL noise

    @show _, shift, lambda = ISOKANN.isotarget(iso.model, iso.data.features[1], iso.data.features[2], iso.transform, shiftscale=true)
    Tmax = stepsize(sim) * steps(sim)
    q = log(lambda) / Tmax
    b = shift
    @assert q <= 0

    χ(x) = ISOKANN.chicoords(iso, x) |> ISOKANN.myonly

    function bias(x; t, sigma)
        λ = exp(q * (Tmax - t))
        logψ(x) = log(λ * (χ(x) - b) + b)
        u = σ .* σ .* only(ISOKANN.Zygote.gradient(logψ, x))
        return forcescale .* u
    end

    return bias
end

using LinearAlgebra  # For pseudoinverse function
function shift_and_scale(xs, ys)
    X = [ones(length(xs)) xs]
    X_pinv = pinv(X)
    β = X_pinv * ys
    bias = β[1]
    scale = β[2]
    limit = bias / (1 - scale)
    return bias, scale, limit
end

function shift_and_scale(iso::ISOKANN.Iso)
    xs = ISOKANN.expectation(iso.model, iso.data.features[1]) |> ISOKANN.cpu
    ys = ISOKANN.expectation(iso.model, iso.data.features[2]) |> ISOKANN.cpu
    shift_and_scale(xs, ys)
end

end#module