using LinearAlgebra: norm
import Zygote
import ISOKANN.OpenMM: trajectory

mutable struct GuidedLangevinBridge
    sim
    xi
    guide
    gain
end

angdiff(ξ, z) = @.(mod(ξ - z + π, 2π) - π)

WITHSIGMA::Bool = true

function biasforce(B::GuidedLangevinBridge; withsigma=WITHSIGMA)
    function bias(x; t, sigma, F)
        J = Zygote.jacobian(B.xi, x) |> only
        G = B.gain(t)
        z = B.guide(t)
        xi = B.xi(x)

        #@show size.((J,G,z,xi, x))
        u = J' * G * angdiff(z,xi)
        #@show x, norm(u)

        if withsigma
            # default scaling of u with sigma happens in girsanov integrator
            return u
        else
            # avoid scaling of u with sigma (canceling the factor in girsanov integrator)
            return u ./ sigma
        end
    end
end

function trajectory(B::GuidedLangevinBridge; x0)
    steps = floor(Int, tmax(B) / OpenMM.stepsize(B.sim))
    OpenMM.integrate_girsanov(B.sim; x0, steps, bias=biasforce(B))
end

##

tmax(B::GuidedLangevinBridge) = maximum(B.guide.xs) # largest argument of the interpolation object

struct LinearInterpolant{TX,TY}
    xs::TX              # AbstractVector
    ys::TY              # AbstractMatrix (n × N)
end

function (itp::LinearInterpolant)(x)
    xs, ys = itp.xs, itp.ys
    i = searchsortedlast(xs, x)
    i = clamp(i, 1, length(xs) - 1)
    t = (x - xs[i]) / (xs[i+1] - xs[i])
    @views (1 - t) * ys[:, i] + t * ys[:, i+1]
end

##

function GuidedLangevinBridge(;
    sim=OpenMMSimulation(step=5e-6, integrator="brownian"),
    g = 1, 
    T=0.1, )
    
    global X0 = ISOKANN.coords(sim)

    ts = [0,T]
    zs = [ZA ZB]

    guide = LinearInterpolant(ts, zs)
    

    gain(t) = g
    xi(x) = [ISOKANN.phi(x), ISOKANN.psi(x)]

    B = GuidedLangevinBridge(sim, xi, guide, gain)
end



using Plots
function plot_guided(;n=5, T=.1, k=10)
    plot()
    xs = []
    gs = []

    @time for _ in 1:n
        B = GuidedLangevinBridge(; g=k, T)
        if k === false
            steps = floor(Int, tmax(B) / OpenMM.stepsize(B.sim))
            x = OpenMM.trajectory(B.sim, steps)
            push!(gs, 0)
            x = x[:, 1:1000:end]
        else
            G = trajectory(B, x0=XA)
            @show G[2]
            push!(gs, G[2])
            x = G[3][:, 1:1000:end]
        end
        push!(xs, x)
        plot!(ISOKANN.phi(x), ISOKANN.psi(x), alpha=0.8, marker=:x)
    end
    ess = 1/sum(g->exp(-g)^2, gs)
    @show ess
    plot!(title = "T = $T, \\kappa=$k", xlims=(-pi, pi), ylims=(-pi, pi)) |> display
    savefig("plots/plot_guided T=$T k=$k.png")
    return reduce(hcat, xs), gs
end

phipsi(x::AbstractVector) = [ISOKANN.phi(x), ISOKANN.psi(x)]
phipsi(x::AbstractMatrix) = mapslices(phipsi, x, dims=1)

const ZA = [-2.5, 2.5]
const ZB = [-1.5, 1]




function findcorestates(xs)
    a = argmin(norm.(eachcol(phipsi(xs) .- ZA)))
    b = argmin(norm.(eachcol(phipsi(xs) .- ZB)))
    xs[:,a], xs[:,b]
end

XA = [0.3315369966853357, 2.5799381302680535, 0.3168867078236774, 0.2451820948007519, 2.6179884144007604, 0.28161079243477155, 0.2427263204707839, 2.7205943585533556, 0.2878471999931951, 0.1632164390727611, 2.5647464316368507, 0.3391467157253451, 0.22881446261934646, 2.5656482329870784, 0.14636854032925964, 0.19145539192755703, 2.6405147402197704, 0.05463216193534928, 0.2441318536564489, 2.436742569785734, 0.12354210202268925, 0.293449168604016, 2.3690082651145983, 0.20326701528328725, 0.20687537267742587, 2.374526004407524, -0.004751292077700521, 0.12679368350225256, 2.4430222621309556, -0.05652804761235346, 0.31806856848305076, 2.3657882807327324, -0.10866414348799437, 0.3996756905011949, 2.301298882129409, -0.06608048079427262, 0.2805339959984766, 2.3201191353644943, -0.20908931743385128, 0.3683753519452983, 2.4592540293662033, -0.12903347488780526, 0.14723924436455807, 2.2337543849548793, 0.02345969291554982, 0.19039095248373292, 2.1600473911974376, 0.1156709987075997, 0.049711745672056815, 2.195980824291773, -0.05476065444136356, 0.029027616504305735, 2.264641838485128, -0.11396514955625085, -0.03933321677849931, 2.0856248510963176, -0.01902543924042347, -0.05531122265271535, 2.019032984985613, -0.10658009129247659, -0.0038123906392522836, 2.014061866052555, 0.06469647369027595, -0.12938204638768008, 2.1275439675958707, 0.004339092478882697]

XB = [0.23214637843610542, 2.6995231218208606, 0.23899360636138597, 0.29503941549654855, 2.6632974548910253, 0.1649843295754245, 0.3813446279195921, 2.6261417145459744, 0.21830028475855945, 0.3227026948486557, 2.7424499696915148, 0.09918357573295923, 0.21402887737379775, 2.5669657316851464, 0.08592156716585772, 0.1013427160669698, 2.5747935569996963, 0.06768023310497485, 0.2857514648074136, 2.4800260611586, 0.022848100225059398, 0.3906241855782488, 2.475819357439268, 0.05182475352804368, 0.23384624970564605, 2.3976861048978266, -0.08326420607709725, 0.148200102160194, 2.443172514990825, -0.12672089487499008, 0.3437363453351647, 2.3619267672791926, -0.18649044639836632, 0.4464476706470792, 2.368707417689508, -0.1443987185839608, 0.33091039315560433, 2.264216470272857, -0.23072054425807753, 0.32372513098138134, 2.447859986020736, -0.27511592768890786, 0.1700156921725329, 2.2656576089776856, -0.02519955870209597, 0.21158988327826242, 2.152808099495209, -0.056919794549844395, 0.07064999499715348, 2.2804569659771143, 0.0633370172912183, 0.03337059988080972, 2.3540582747426786, 0.07284292242095171, 0.023771262652676248, 2.1723041684926363, 0.15466662419391894, -0.04563166834811809, 2.0972957847560063, 0.09816069373771923, 0.11217880965353097, 2.1115307431131933, 0.18414211335059794, -0.02842774097806951, 2.2101784988324176, 0.25187838064465745]