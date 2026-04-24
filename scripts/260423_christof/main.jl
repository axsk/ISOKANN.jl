using NPZ
using ISOKANN
using LinearAlgebra
using Interpolations
using Zygote
using ProgressMeter
using JLD2
#include("../251126_carsten/api_adp_grid.jl")

x0 = npzread("AMORE/X0_metad.npy")' # samples with metadynamics in phi-psi, shape (66, 160000)
#xt = npzread("AMORE/Xtau_metad.npy")'

STEPS = 2500
sim = OpenMMSimulation(steps=STEPS, temp=450) # 5ps

x0bins = [[] for _ in 1:length(GRID)^2]
for x in eachcol(x0)
    push!(x0bins[boxindex(x)], x)
end

""" starting in each box, compute NK tau-propagated samples
return xt of shape (66, NK, nboxes) """
function compute_xt(;nk=100)
    xt = zeros(66, nk, length(x0bins))
    @showprogress for (i, bin) in enumerate(x0bins)
        isempty(bin) && (@warn("No starting point in bin $i"); continue)
        for k in 1:nk
            xt[:, k, i] = trajectory(sim; x0=rand(bin), saveevery=OpenMM.steps(sim))[:, end]
        end
    end
    global XT = xt
    jldsave("xt.jld2"; xt)
    return xt
end

""" grid utilities """

GRID = range(-pi, pi, 40+1)[1:end-1]

phipsi(x) = (ISOKANN.phi(x), ISOKANN.psi(x))

function gridindex(angle::Number)
    n = length(GRID)
    Δ = 2π / n
    mod(floor(Int, (angle - GRID[1] + Δ / 2) / Δ), n) + 1
end

function boxindex(x::Tuple)
    @assert length(x) == 2
    n = length(GRID)
    inds = LinearIndices((n, n))
    i1 = gridindex(x[1])
    i2 = gridindex(x[2])
    return inds[i1, i2]
end

boxindex(x::AbstractArray) = boxindex(phipsi(x))

function transfermatrix(xt, eps=0)
    n = length(GRID)
    C = zeros(n^2, n^2) .+ 0
    nkoop, nsamples = size(xt, 2), size(xt, 3)
    for i in 1:nsamples
        for k in 1:nkoop
            j = boxindex(xt[:, k, i])
            C[i, j] += 1
        end
    end

    T = C ./ sum(C, dims=2)
    return T
end



P = transfermatrix(x0,xt)
@show reverse(eigen(P).values)[1:3]

function neighborhood(φ, ψ, r=5)
    n = length(GRID)
    inds = LinearIndices((n, n))
    i0 = gridindex(φ)
    j0 = gridindex(ψ)
    [inds[mod(i0 + di - 1, n) + 1, mod(j0 + dj - 1, n) + 1] for di in -r:r, dj in -r:r] |> vec
end

phiB, psiB = deg2rad.((-108 + 180, 157.5 - 180))
setB = neighborhood(phiB, psiB, 5)

indB = falses(length(GRID)^2)
indB[setB] .= true

PB = P^20 * indB

phi0, psi0 = deg2rad.((103.5 - 180, 148.5 - 180))
indx0 = boxindex((phi0, psi0))
pb_t = @show PB[indx0]

x0s = x0list[indx0]

function rama_heat(x)
    heatmap(GRID, GRID, x', aspect_ratio=1, xlabel="phi", ylabel="psi", xlims=(-π, π), ylims=(-π, π))
end


function estimate_pb_mc(;n=100, Tmax = 100)
    manysteps = round(Int, Tmax / OpenMM.stepsize(sim))
    countB = 0
    xts = []
    @showprogress for i in 1:n
        x0 = rand(x0s)
        xt = trajectory(sim, manysteps; x0, saveevery=manysteps)[:,end]
        push!(xts, xt)
        if boxindex(xt) in setB
            println("hit B")
            countB += 1
        end
    end
    return countB / n, xts
end

# wilson 95% interval

### controlled dynamics
U_add(phi, psi) = (1-cos(psi-psi0) + 1 + cos(phi - phi0))^4


PB_k = [P^(20 - k + 1) * indB for k in 1:20]

interpolate_rama_cubic(X) = extrapolate(
    scale(interpolate(X, BSpline(Cubic(Periodic(OnCell())))), GRID, GRID),
    Periodic(),
)

interpolate_rama_linear(X) = extrapolate(scale(interpolate(X, BSpline(Linear())), GRID, GRID),Periodic(), )


PB_ε = 1e-3 * maximum(maximum.(PB_k))  # regularization avoiding log(0) singularity, minor impact where PB actually has mass
logPB_itp = [interpolate_rama_linear(log.(reshape(p, 40, 40) .+ PB_ε)) for p in PB_k]

function bias(x; t, kwargs...)
    nk = length(logPB_itp)
    τ = OpenMM.stepsize(sim) * OpenMM.steps(sim)   # duration per PB slice
    k = clamp(floor(Int, t/τ) + 1, 1, nk)

    dlogpdx = Zygote.gradient(x) do x
        φ, ψ = phipsi(x)
        logPB_itp[k](φ, ψ) #+ U_add(φ, ψ)
    end |> only

    # questionable point: which sigma here? here we have ODL*UDL = 2/beta
    sigma_ODL= OpenMM.constants(sim, true).sigma
    return sigma_ODL .* dlogpdx
end

function guided_traj()
    x0 = rand(x0s)
    steps = OpenMM.steps(sim) * length(PB_itp)
    ws = ISOKANN.OpenMM.langevin_girsanov!(sim; x0, bias, steps, sigmascaled=true)
    scatter_ramachandran(ws.values) |> display
    return ws.weights[end], ws.values[:, end]
end

# girsanov estimator for E[1_B(x_T)]
acc = 0
den = 0
for i in 1:500
    w, x = guided_traj()
    if boxindex(x) in setB
        acc += w
    end
    den += w
end
pb_guided = acc / den