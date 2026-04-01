using Flux, Zygote
using ISOKANN.OpenMM: integrate_girsanov
using ISOKANN: coords
using ISOKANN
using LinearAlgebra: norm, dot
using Plots

struct MetadynamicsState
    centers::Vector{Vector{Float32}}   # visited RC values s_i
    height::Float32                     # h
    sigma::Float32                      # σ
end


global dT = 600
global HEIGHT = 1f0
global SIGMA = 0.1f0

MetadynamicsState(; height=HEIGHT, sigma=SIGMA) =
    MetadynamicsState(Vector{Float32}[], height, sigma)

# Bias potential in RC space: V(s) = sum_i h * exp(-|s-s_i|^2 / 2σ^2)
function bias_potential(meta::MetadynamicsState, z::AbstractVector)
    global dT
    isempty(meta.centers) && return 0f0
    @assert length(z) == length(meta.centers[1])
    wN = sum(meta.centers) do sᵢ
        meta.height * exp(-sum(abs2, z .- sᵢ) / (2 * meta.sigma^2))
    end
    ## classic metadynamics:
    isfinite(dT) || return wN # classic MD
    return dT * log(1+ wN / dT) # well tempered MD
end

# Bias force in configuration space: -dV/dx = -dV/ds * ds/dx
function make_bias(iso, meta::MetadynamicsState)
    function bias(x::AbstractVector; kwargs...)
        grad, = Zygote.gradient(x) do x_
            z = chicoords(iso, x_)
            if false && length(z) > 1 # experimental projection
                u = ones(length(z))
                u = u ./ norm(u)
                z = z - dot(z, u) * u
            end
            bias_potential(meta, z)
        end
        return -grad   # negative gradient = force
    end
    return bias
end


mutable struct MetadynamicsSimulator
    sim
    iso
    force
    meta
end

function plot_profile(mdsim::MetadynamicsSimulator, zs=-0.1:0.01:1.1)
    T = OpenMM.temp(mdsim.sim)
    global dT
    if ISOKANN.outputdim(mdsim.iso.model) == 1
        V = [bias_potential(mdsim.meta, [z]) for z in zs]
        F = -(T + dT) / T * V
        plot(zs, F, xlabel="z", ylabel="V(z)", title="Metadynamics Free Energy Profile")
        chilast = chicoords(mdsim.iso, coords(mdsim.iso)[:, end]) |> cpu
        Flast = -(T + dT) / T * bias_potential(mdsim.meta, chilast)
        scatter!(chilast, [Flast])
    else

        V = [bias_potential(mdsim.meta, x) for x in eachcol(chis(mdsim.iso))]
        F = -(T + dT) / T * V
        scatter(eachrow(chis(mdsim.iso))..., marker_z=F, camera=(135, 35))
    end
end

function MetadynamicsSimulator(iso; height=HEIGHT, sigma=SIGMA)
    iso = cpu(iso)
    sim = iso.data.sim
    meta = MetadynamicsState(eachcol(chis(iso)), height, sigma)
    bias = make_bias(iso, meta)
    return MetadynamicsSimulator(sim, iso, bias, meta)
end

trajectory(mdsim::MetadynamicsSimulator; kwargs...) = OpenMM.langevin_girsanov!(mdsim.sim; bias=mdsim.force, sigmascaled=false, kwargs...)

#cind = 0
 
function adaptive_metadynamics(iso; deposit=OpenMM.steps(iso.data.sim), height=HEIGHT, sigma=SIGMA, x0=coords(iso)[:, end])
    md=MetadynamicsSimulator(iso; height, sigma)
    @time "Girsanov trajectory" t = trajectory(md; x0, saveevery=deposit) 
    @assert norm(t.values[:, end]) < 100
    #global cind += 1
    #scatter_ramachandran(values(t), marker_z=cind) |> display
    
    #scatter_ramachandran(iso) |> display
    xnew = values(t)
    @show t.weights
    addcoords!(iso, xnew)
    return (;t, md, xnew)
end


XA = [0.33898000519238475, 2.581167554816886, 0.316137526467131, 0.24439430356143435, 2.618430843485813, 0.2812859911489418, 0.23998302166311014, 2.7279220679365097, 0.285368352737877, 0.16468290582358056, 2.570972975387856, 0.33198123160123, 0.22857214798948508, 2.565362312254419, 0.14579321188206523, 0.19124136416905824, 2.640726157362197, 0.0541026851238735, 0.24402186183692884, 2.4376353440711793, 0.12412232349973612, 0.2903040092295479, 2.3732548452348246, 0.19469102102705307, 0.20621933292058534, 2.374440344600457, -0.005241982629152157, 0.12896230297352781, 2.4335152274693472, -0.05469143502399562, 0.3174674811534015, 2.3648349336534156, -0.1093092348999846, 0.39698603724050746, 2.305809653736958, -0.0691791772900721, 0.28200588004971566, 2.322397147776998, -0.20504393458777626, 0.365360125658866, 2.462250943604863, -0.12824506472062086, 0.14862971899677796, 2.2345166995172963, 0.022383994166212888, 0.19080588347057845, 2.1601304004213957, 0.11636374234582904, 0.04922629582794467, 2.1960230097429876, -0.05394814441600838, 0.020682229122924736, 2.2657197378660245, -0.11964193675959689, -0.03899663770750985, 2.0845726886455003, -0.020187349765587764, -0.04861489411580564, 2.0202850988240058, -0.10559863641880232, -0.0073833158057770845, 2.014701958301892, 0.06459809053967074, -0.1297626878829161, 2.1326462424993893, 0.006689341363523314]
XB = [0.27860354484813593, 2.2563790191252573, 0.49030072062041363, 0.3708465230992061, 2.2322931055168116, 0.43471792087636935, 0.3854221073489557, 2.1222610257145273, 0.44451257584753306, 0.44991288100325116, 2.2935514349495123, 0.47634573480900955, 0.3477693483404822, 2.2591800330057827, 0.28569881088333365, 0.247031321511723, 2.3168173767333298, 0.2488662624944461, 0.4443914233189699, 2.2280836867848532, 0.20109622759917076, 0.5164354414275171, 2.16687616385289, 0.23419934562842715, 0.46239591470232844, 2.290752975531989, 0.07025758773107382, 0.4118601698360281, 2.3819509677913633, 0.060086239089671874, 0.6085303068163623, 2.3062585881658086, 0.03550846605911172, 0.645129282916172, 2.2064232160210566, 0.015192703546145018, 0.6200794950345564, 2.359928824811465, -0.052711599698312905, 0.675069878683435, 2.3623609404856976, 0.10920900759141564, 0.3740347205846337, 2.2178994354984694, -0.038645802054178904, 0.42691022272268425, 2.18188229679336, -0.14546824914497522, 0.24505795181670825, 2.2122667992815637, -0.016351126739953196, 0.2158613626468009, 2.2418276684219425, 0.07484394183321985, 0.1517083469747925, 2.1594692245047695, -0.12034757695583856, 0.057957813692806316, 2.2074006034926943, -0.12636083723730407, 0.2011382993471141, 2.1542625080629003, -0.21617649397685243, 0.13742549149129366, 2.052776150372571, -0.08252540831102302]
XC = [0.45542977606928714, 2.2920101447835832, 0.45145048358190043, 0.36035934720912133, 2.249671003140913, 0.4160704684660634, 0.2762848267135616, 2.2874756545987114, 0.47129307736255327, 0.3565638313229723, 2.142148050686486, 0.4305569807729916, 0.32845961588603023, 2.2744138098586117, 0.2686251408232698, 0.22061818833862093, 2.240186696585767, 0.23068723053650278, 0.4255033736343527, 2.3300977527442024, 0.19659145860936372, 0.5052506638275116, 2.3597189053779077, 0.24793662984842404, 0.4637848321308562, 2.2951824742729565, 0.05698049060032198, 0.5221625108686456, 2.372402705915494, 0.017732779095097108, 0.5640155034819058, 2.1772874319788755, 0.058048325606688614, 0.5376573081076875, 2.0879362146433085, 0.11454799875116117, 0.5887925301722823, 2.154941830056006, -0.04168095295711932, 0.6454302932098361, 2.214756388477496, 0.11753070010327221, 0.3575007276125249, 2.2858458433287954, -0.05026951604988611, 0.3829290133220176, 2.3386072696516207, -0.1549403066134884, 0.25595537269646856, 2.203617656588123, -0.03525394505777248, 0.23406585702056068, 2.1708007892431, 0.059517472249794964, 0.15174351816766254, 2.1820742098086674, -0.13193194523812926, 0.10256884106963207, 2.079481744065602, -0.12109002848798586, 0.06967059989778537, 2.248123452199889, -0.10299169124981404, 0.17715600758393196, 2.19356316536616, -0.23784840560899925]

function geniso(;nk=8, steps=100, nd=1, kwargs...)
    # using longer lag because i feared that the bias would move through space on a "thin" trajectory
    # which in turn could lead to the single-markov-chain problem of each sample just connecting to the following
    # using a longer lag regularizes dynamically (as opposed to say through the nn which regularizes spatially)
    # this conjecture needs further investigation
    sim = OpenMMSimulation(;steps)
    # i am concerned that the hydrogen rotation might be the slowest process, so i am ignoring all hydrogens for the features
    featurizer = OpenMM.FeaturesAtoms([i for (i, a) in enumerate(OpenMM.atoms(sim)) if string(a.element.symbol) != "H"])
    # starting with *very* few samples
    xs = hcat(XA,XB,XC)
    data = SimulationData(sim, xs, 8; featurizer)
    iso = Iso(data, model=ISOKANN.densenet(layers=[45, 20, 20, 20, nd], layernorm=true), opt=AdamRegularized(1e-3,1e-5); kwargs...)
end

function run_metadynamics!(iso; generations=100, samples=10, plots=[], height=HEIGHT, sigma=SIGMA, iter=100)
    for _ in 1:generations
        @time adaptive_metadynamics(iso; height, sigma); 
        @time run!(iso, iter); 
        if plots != false
            l = @layout [[a; b] c{0.3w} ]
            global p1=scatter_ramachandran(iso); 
            scatter!(p1, [ISOKANN.phi(coords(iso)[:, end])], [ISOKANN.psi(coords(iso)[:, end])])
            global p2=plot_training(iso);
            global p3=plot_profile(MetadynamicsSimulator(iso; height, sigma))
            p=plot(p1,p3,p2, layout=l, size=(800,800))
            display(p)
            push!(plots, p)
        end
    end
    return (;iso, plots)
end

function run_kde_dash!(iso; generations=1, plots=[], kwargs...)
    for _ in 1:generations
        ISOKANN.run_kde!(iso; generations=1, kwargs...)
        global p1 = scatter_ramachandran(iso)
        global p2 = plot_training(iso)
        p = plot(p1, p2, layout=(1, 2), size=(800, 800))
        push!(plots, p)
    end
    return plots
end

function run_both!(iso; generations=100, samples_md=1, samples_kde=1, iter=100, plots=[])
    for i in 1:generations
        runadaptive!(iso; generations=1, kde=samples_kde, iter)
        run_metadynamics!(iso; generations=1, samples=samples_md, iter, plots)
    end
end

function test_metadynamics(iters=100)
    iso = geniso()
    global ISO = iso
    try 
        for _ in 1:iters
            adaptive_metadynamics(iso)
            #run!(iso, 100)
            runadaptive!(iso, generations=1, kde=10, iter=100)
        end
    catch e
        !(e isa InterruptException) && rethrow(e)
    end
    return iso
end

function reactivepath_save(iso;kwargs...)
    i = hash(kwargs)
    @show i
    ix=save_reactive_path(iso, sigma=.03, minjump=0.02, maxjump=0.1, out="experiments/260311 metadyn adp/path$i.pdb"; kwargs...)
    OpenMM.potential(sim,coords(iso)[:,ix])|>vec|>plot; savefig("experiments/260311 metadyn adp/energy$i.png");
    scatter_ramachandran(coords(iso)[:, ix], x->chicoords(iso, x)); savefig("experiments/260311 metadyn adp/rama$i.png")
end

function makeanim(ps, filename)
    a=Animation()
    for p in ps
        frame(a, p)
    end
    mp4(a, filename)
end