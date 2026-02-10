using LinearAlgebra: norm
import Zygote
import ISOKANN.OpenMM: trajectory
using ISOKANN: coords

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
        u = J' * G * angdiff(z,xi)  # TODO: angdiff is only used for periodic RCs
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

function brigde_rama(;
    sim=OpenMMSimulation(friction=1000, integrator="brownian"),
    g = 1, 
    T=1,
    z0 = ZA, 
    z1 = ZB)
    
    ts = [0,T]
    zs = [z0 z1]

    guide = LinearInterpolant(ts, zs)
    

    gain(t) = g
    xi(x) = [ISOKANN.phi(x), ISOKANN.psi(x)]

    GuidedLangevinBridge(sim, xi, guide, gain)
end

function isobridge(iso::Iso; g=.3, T=20, extrapolate=0.)

    gd = guide(iso, T; extrapolate)
    gain(t) = g
    xi(x) = chicoords(iso, x)

    GuidedLangevinBridge(sim, xi, gd, gain)
end

function guide(iso::Iso, Tmax; extrapolate=0.)
    a = 0 - extrapolate
    b = 1 + extrapolate
    zs = only(chicoords(iso, XA)) < only(chicoords(iso, XB)) ? [a, b] : [b, a]
    LinearInterpolant([0,Tmax], zs')
end


using Plots
# time unit is picoseconds
function plot_guided(;n=1, T=10, g=1)
    plot()
    xss = Matrix{Float64}[]
    logws = []

    for _ in 1:n
        B = GuidedLangevinBridge(; g, T)
        if g === false
            steps = floor(Int, tmax(B) / OpenMM.stepsize(B.sim))
            xs = OpenMM.trajectory(B.sim, steps)
            push!(logws, 0)
        else
            _, logw, xs = trajectory(B, x0=XA)
            push!(logws, @show logw)
        end
        push!(xss, xs[:, 1:100:end])
        zs = phipsi(xs[:, 1:100:end])
        plot!(eachrow(zs)..., alpha=0.8, marker=:x)
    end
    ws = @. exp(logws)
    ws ./= sum(ws)
    ess = 1/sum(ws.^ 2)
    @show ess
    plot!(title = "T = $T, g=$g", xlims=(-pi, pi), ylims=(-pi, pi)) |> display
    scatter!(eachrow([ZA ZB])...)
    savefig("plots/plot_guided T=$T g=$g sig=$WITHSIGMA.png")
    return xss, logws
end

phipsi(x) = vcat(ISOKANN.phi(x)', ISOKANN.psi(x)')



function addpaths!(;paths=3, iso=iso, G=G, train=paths*1000)
    scatter_ramachandran(iso)
    for i in 1:paths
        @time t = trajectory(G;x0=XB)
        addcoords!(iso, t[3][:, 1:30:end])
        plot!(eachrow(phipsi(t[3]))..., alpha=0.8)
    end
    plot!() |> display
    run!(iso, train)
end



function findcorestates(xs)
    a = argmin(norm.(eachcol(phipsi(xs) .- ZA)))
    b = argmin(norm.(eachcol(phipsi(xs) .- ZB)))
    xs[:,a], xs[:,b]
end

#XA = [0.3315369966853357, 2.5799381302680535, 0.3168867078236774, 0.2451820948007519, 2.6179884144007604, 0.28161079243477155, 0.2427263204707839, 2.7205943585533556, 0.2878471999931951, 0.1632164390727611, 2.5647464316368507, 0.3391467157253451, 0.22881446261934646, 2.5656482329870784, 0.14636854032925964, 0.19145539192755703, 2.6405147402197704, 0.05463216193534928, 0.2441318536564489, 2.436742569785734, 0.12354210202268925, 0.293449168604016, 2.3690082651145983, 0.20326701528328725, 0.20687537267742587, 2.374526004407524, -0.004751292077700521, 0.12679368350225256, 2.4430222621309556, -0.05652804761235346, 0.31806856848305076, 2.3657882807327324, -0.10866414348799437, 0.3996756905011949, 2.301298882129409, -0.06608048079427262, 0.2805339959984766, 2.3201191353644943, -0.20908931743385128, 0.3683753519452983, 2.4592540293662033, -0.12903347488780526, 0.14723924436455807, 2.2337543849548793, 0.02345969291554982, 0.19039095248373292, 2.1600473911974376, 0.1156709987075997, 0.049711745672056815, 2.195980824291773, -0.05476065444136356, 0.029027616504305735, 2.264641838485128, -0.11396514955625085, -0.03933321677849931, 2.0856248510963176, -0.01902543924042347, -0.05531122265271535, 2.019032984985613, -0.10658009129247659, -0.0038123906392522836, 2.014061866052555, 0.06469647369027595, -0.12938204638768008, 2.1275439675958707, 0.004339092478882697]
XA = [0.33898000519238475, 2.581167554816886, 0.316137526467131, 0.24439430356143435, 2.618430843485813, 0.2812859911489418, 0.23998302166311014, 2.7279220679365097, 0.285368352737877, 0.16468290582358056, 2.570972975387856, 0.33198123160123, 0.22857214798948508, 2.565362312254419, 0.14579321188206523, 0.19124136416905824, 2.640726157362197, 0.0541026851238735, 0.24402186183692884, 2.4376353440711793, 0.12412232349973612, 0.2903040092295479, 2.3732548452348246, 0.19469102102705307, 0.20621933292058534, 2.374440344600457, -0.005241982629152157, 0.12896230297352781, 2.4335152274693472, -0.05469143502399562, 0.3174674811534015, 2.3648349336534156, -0.1093092348999846, 0.39698603724050746, 2.305809653736958, -0.0691791772900721, 0.28200588004971566, 2.322397147776998, -0.20504393458777626, 0.365360125658866, 2.462250943604863, -0.12824506472062086, 0.14862971899677796, 2.2345166995172963, 0.022383994166212888, 0.19080588347057845, 2.1601304004213957, 0.11636374234582904, 0.04922629582794467, 2.1960230097429876, -0.05394814441600838, 0.020682229122924736, 2.2657197378660245, -0.11964193675959689, -0.03899663770750985, 2.0845726886455003, -0.020187349765587764, -0.04861489411580564, 2.0202850988240058, -0.10559863641880232, -0.0073833158057770845, 2.014701958301892, 0.06459809053967074, -0.1297626878829161, 2.1326462424993893, 0.006689341363523314]

#XB = [0.23214637843610542, 2.6995231218208606, 0.23899360636138597, 0.29503941549654855, 2.6632974548910253, 0.1649843295754245, 0.3813446279195921, 2.6261417145459744, 0.21830028475855945, 0.3227026948486557, 2.7424499696915148, 0.09918357573295923, 0.21402887737379775, 2.5669657316851464, 0.08592156716585772, 0.1013427160669698, 2.5747935569996963, 0.06768023310497485, 0.2857514648074136, 2.4800260611586, 0.022848100225059398, 0.3906241855782488, 2.475819357439268, 0.05182475352804368, 0.23384624970564605, 2.3976861048978266, -0.08326420607709725, 0.148200102160194, 2.443172514990825, -0.12672089487499008, 0.3437363453351647, 2.3619267672791926, -0.18649044639836632, 0.4464476706470792, 2.368707417689508, -0.1443987185839608, 0.33091039315560433, 2.264216470272857, -0.23072054425807753, 0.32372513098138134, 2.447859986020736, -0.27511592768890786, 0.1700156921725329, 2.2656576089776856, -0.02519955870209597, 0.21158988327826242, 2.152808099495209, -0.056919794549844395, 0.07064999499715348, 2.2804569659771143, 0.0633370172912183, 0.03337059988080972, 2.3540582747426786, 0.07284292242095171, 0.023771262652676248, 2.1723041684926363, 0.15466662419391894, -0.04563166834811809, 2.0972957847560063, 0.09816069373771923, 0.11217880965353097, 2.1115307431131933, 0.18414211335059794, -0.02842774097806951, 2.2101784988324176, 0.25187838064465745]
XB = [0.27860354484813593, 2.2563790191252573, 0.49030072062041363, 0.3708465230992061, 2.2322931055168116, 0.43471792087636935, 0.3854221073489557, 2.1222610257145273, 0.44451257584753306, 0.44991288100325116, 2.2935514349495123, 0.47634573480900955, 0.3477693483404822, 2.2591800330057827, 0.28569881088333365, 0.247031321511723, 2.3168173767333298, 0.2488662624944461, 0.4443914233189699, 2.2280836867848532, 0.20109622759917076, 0.5164354414275171, 2.16687616385289, 0.23419934562842715, 0.46239591470232844, 2.290752975531989, 0.07025758773107382, 0.4118601698360281, 2.3819509677913633, 0.060086239089671874, 0.6085303068163623, 2.3062585881658086, 0.03550846605911172, 0.645129282916172, 2.2064232160210566, 0.015192703546145018, 0.6200794950345564, 2.359928824811465, -0.052711599698312905, 0.675069878683435, 2.3623609404856976, 0.10920900759141564, 0.3740347205846337, 2.2178994354984694, -0.038645802054178904, 0.42691022272268425, 2.18188229679336, -0.14546824914497522, 0.24505795181670825, 2.2122667992815637, -0.016351126739953196, 0.2158613626468009, 2.2418276684219425, 0.07484394183321985, 0.1517083469747925, 2.1594692245047695, -0.12034757695583856, 0.057957813692806316, 2.2074006034926943, -0.12636083723730407, 0.2011382993471141, 2.1542625080629003, -0.21617649397685243, 0.13742549149129366, 2.052776150372571, -0.08252540831102302]
const ZA = [-2.5, 2.5]
const ZB = [-1.5, 1]
const ZC = [1, -1]

XC = [0.45542977606928714, 2.2920101447835832, 0.45145048358190043, 0.36035934720912133, 2.249671003140913, 0.4160704684660634, 0.2762848267135616, 2.2874756545987114, 0.47129307736255327, 0.3565638313229723, 2.142148050686486, 0.4305569807729916, 0.32845961588603023, 2.2744138098586117, 0.2686251408232698, 0.22061818833862093, 2.240186696585767, 0.23068723053650278, 0.4255033736343527, 2.3300977527442024, 0.19659145860936372, 0.5052506638275116, 2.3597189053779077, 0.24793662984842404, 0.4637848321308562, 2.2951824742729565, 0.05698049060032198, 0.5221625108686456, 2.372402705915494, 0.017732779095097108, 0.5640155034819058, 2.1772874319788755, 0.058048325606688614, 0.5376573081076875, 2.0879362146433085, 0.11454799875116117, 0.5887925301722823, 2.154941830056006, -0.04168095295711932, 0.6454302932098361, 2.214756388477496, 0.11753070010327221, 0.3575007276125249, 2.2858458433287954, -0.05026951604988611, 0.3829290133220176, 2.3386072696516207, -0.1549403066134884, 0.25595537269646856, 2.203617656588123, -0.03525394505777248, 0.23406585702056068, 2.1708007892431, 0.059517472249794964, 0.15174351816766254, 2.1820742098086674, -0.13193194523812926, 0.10256884106963207, 2.079481744065602, -0.12109002848798586, 0.06967059989778537, 2.248123452199889, -0.10299169124981404, 0.17715600758393196, 2.19356316536616, -0.23784840560899925]

function pickfeatures!(iso, n)
    pick = ISOKANN.picking(features(iso.data), n)
    iso.data = iso.data[pick.is]
end

function loss_res(iso, model=iso.model, reg=1e-3)
    chi = model(features(iso))
    kchi = ISOKANN.expectation(model, propfeatures(iso))

    K = chi * pinv(kchi)
    norm(kchi - K * chi) / size(features(iso), 2) + reg * norm(chi * chi'-I)
end

using ProgressMeter

function run_res!(iso, n=1, reg=1e-1)
    @showprogress for i in 1:n
        g = Flux.gradient(m -> loss_res(iso, m, reg), iso.model)
       Flux.update!(iso.opt, iso.model, g[1])
        push!(iso.losses, loss_res(iso, iso.model, reg))
    end
end

using Plots
global XHIST = reshape(XC, :, 1)
function livetraj(; x0=XC, n=1, lag=1000, batch=10, X=XHIST)
    for i in 1:n
        step = max(1, floor(Int, size(X, 2) / 100))
        x0 = XHIST[:, end]
        @time t = trajectory(sim, batch*lag, saveevery=lag, throw=true; x0)
        X = hcat(X, t)
        global XHIST = X
        Y = hcat(@view(X[:, 1:step:end-100]), X[:, max(1,size(X,2)-100):end])
        plot(eachrow(phipsi(Y))..., title=lag*size(X, 2)) |> display
    end
end

global TS=zeros(66,0)

function addbridges!(iso, n=100; zs = [ZA, ZB, ZC]) 
    xs = map(zs) do z
        is = findall(norm.(eachcol(phipsi(coords(iso)) .- z)) .< 0.1)
        i = rand(is)
        return coords(iso)[:,i]
    end
    ts = reduce(hcat, bridges(; zs, xs))
    ts = ts[:, sort(rand(1:size(ts, 2), n))]
    addcoords!(iso, ts)

end

function bridges(; 
    zs = [ZA, ZB, ZC],
    xs = [XA, XB, XC]
)
    
    ts = []
    ij = [(i,j) for  i in 1:3, j in 1:3 if i!=j]
    for (i,j) in ij
        @show i,j
        b = brigde_rama(;z0=zs[i], z1=zs[j], T=20, g=.3)
        t = trajectory(b; x0=xs[i])[3]
        @show norm(phipsi(t[:,end]) - zs[j])
        push!(ts,t)
        global TS = [TS t]
    end
    return ts
end

allcoords(iso) = hcat(ISOKANN.coords(iso), propcoords(iso) |> ISOKANN.flattenlast)
allfeats(iso) = hcat(features(iso), propfeatures(iso) |> ISOKANN.flattenlast)

function resample_picking_features!(iso, n=length(iso.data))
    #ys = propcoords(iso) |> ISOKANN.flattenlast
    #fys = propfeatures(iso) |> ISOKANN.flattenlast
    ys = allcoords(iso)
    fys = allfeats(iso)
    is = picking(fys, n).is
    is = sort(is)
    xs = ys[:, picking(fys, n).is]
    iso.data = similar(iso.data, xs)
end


Base.similar(data::SimulationData, xs) = SimulationData(data.sim, xs, ISOKANN.nk(data), featurizer=data.featurizer)
function ISOKANN.SimulationData(data::ISOKANN.SimulationData; featurizer)
    SimulationData(data.sim, (ISOKANN.coords(data), ISOKANN.propcoords(data)); featurizer)
end

function adaptive(n)
    for _ in 1:n
        @time "picking" resample_picking_features!(iso, 2000)
        @time "bridges" addbridges!(iso, 200)
        @time "kde" resample_kde!(iso, 200)
        run!(iso, 3000)
    end
end

# TODO if density  is repsponsible, pick by ramach angle