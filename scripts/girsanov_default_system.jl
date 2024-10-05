using ISOKANN
using CUDA

using ForwardDiff

g = (x, y) -> zeros(size(x,1))

sim = OpenMMSimulation(
    steps=10000,
    temp=25 + 272.15,
    featurizer=x -> ISOKANN.flatpairdists(x),
    minimize=false,
    gpu=CUDA.functional();
    F_ext= g
)


function control(x, t, T, σ, χ, q, b, forcescale)
    forcescale == 0. && return zero(x)
    @assert q <= 0
    t>T && (t=T)
    @assert t<=T

    λ = exp(q * (T-t))
    if λ*(χ(x)-b) + b <= 0
        @show χ(x), λ, b
        @assert χ(x) > 0
    end
    logψ(x) = log(λ*(χ(x)-b) + b)

    u = forcescale .* ForwardDiff.gradient(logψ, x)
    return u
end

function adddata(d::SimulationData, model, n, u; keepedges=false)
    n == 0 && return d
    xs = ISOKANN.chistratcoords(d, model, n; keepedges)
    addcoordsControl(d, xs, u)
end

function addcoordsControl(d::SimulationData, coords::AbstractMatrix, u)
    mergedata(d, SimulationData(d.sim, coords, ISOKANN.nk(d), featurizer=d.featurizer, u=u))
end
opt = ISOKANN.NesterovRegularized(1e-3, 1e-3)
iso = Iso(sim, nx=10, nk=1; opt=opt)
generations = 3
for g in 1:generations
    s, l = run!(iso, 500; optctrl=true)
    cpumodel = cpu(iso.model)
    u = (x, sigma) -> control(x, 0.002, 1, sigma, y -> abs(first(cpumodel(iso.data.featurizer(y)))), min(log(l), 0), s, 0.1)    
    @time iso.data = adddata(iso.data, iso.model, 4, u)
    plot_training(iso)
end