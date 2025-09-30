# proof of concept for a live visualization of the training on molecular level
# needs a cleaner rewrite. Curren
module MakieExt

using Bonito
using WGLMakie
using WGLMakie.Observables: throttle
using ThreadPools
using CUDA
using ISOKANN
using LinearAlgebra: normalize

USEGPU = CUDA.functional()
ISO = nothing
ISRUNNING = false


function content(session)
    global SESSION
    SESSION = session
    println("session created $session")
    @show Threads.threadid()
    isoui, isoo = @time isocreator()
    global ISO
    ISO = isoo
    #Grid(isoui)
    Grid(isoui, @lift(dashboard($isoo, session)))
end

logrange(x1, x2, n) = round.((10^y for y in range(log10(x1), log10(x2), length=n)), sigdigits=1)

function isocreator()
    pdb = Dropdown(["data/alanine-dipeptide-nowater.pdb", "data/vgv.pdb", "data/villin nowater.pdb", "temp/8eftstepped.pdb"])
    pdbid = TextField("")
    steps = StylableSlider(1:1000, value=10)
    temperature = StylableSlider(-10:70, value=30)
    optim = Dropdown(["Adam", "Nesterov"], index=1)
    learnrate = StylableSlider(sort([logrange(1e-4, 1e-2, 10); 1e-3]), value=1e-3)
    regularization = StylableSlider(sort([logrange(1e-6, 1e-3, 10); 1e-4]), value=1e-4)
    nx = StylableSlider(2:100, value=10)
    nk = StylableSlider(1:10, value=2)


    button = Button("Create")

    global ISRUNNING

    function create_iso()
        opt = if optim.value[] == "Adam"
            ISOKANN.AdamRegularized(learnrate.value[], regularization.value[])
        elseif optim.value[] == "Nesterov"
            ISOKANN.NesterovRegularized(learnrate.value[], regularization.value[])
        else
            error()
        end

        ISRUNNING = true

        getpdb(id::String) = Base.download("https://files.rcsb.org/download/$id.pdb")

        pdbfile = pdbid.value[] != "" ? getpdb(pdbid.value[]) : pdb.value[]


        iso = Iso(
            OpenMMSimulation(
                steps=steps.value[],
                pdb=pdbfile,
                forcefields=ISOKANN.OpenMM.FORCE_AMBER,
                temp=temperature.value[] + 272.15,
                gpu=USEGPU,
                features=0.5,),
            loggers=[], opt=opt, nx=nx.value[], nk=nk.value[], minibatch=64, gpu=USEGPU)

        run!(iso)
        ISRUNNING = false
        return iso
    end


    isoo = Observable(create_iso())

    #on(learnrate) do val
    #    ISOKANN.Flux.adjust!(iso.opt, eta=val)
    #end

    #on(regularization) do val
    #    ISOKANN.Flux.adjust!(iso.opt, lambda=val)
    #qend



    on(button.value) do _
        if !ISRUNNING

            button.attributes[:disabled] = true
            isoo[] = create_iso()
            button.attributes[:disabled] = false

        end
    end

    return Card(Grid(
            pdb, pdbid,
            steps, Bonito.Label(steps.value),
            temperature, Bonito.Label(temperature.value),
            optim, nothing,
            learnrate, Bonito.Label(learnrate.value),
            regularization, Bonito.Label(regularization.value),
            nx, Bonito.Label(nx.value),
            nk, Bonito.Label(nk.value),
            button,
            columns="1fr min-content",
            justify_content="begin",
            align_items="center",
        ); width="300px",), isoo
end


function serve()
    app = App(title="ISOKANN Dashboard") do session
        return content(session)
    end

    server = Bonito.get_server()  # default port is 9384
    route!(server, "/" => app)
    @show server
    return app
end



function bondids(pysim)
    ids = Int[]
    for bond in pysim.topology.bonds()
        push!(ids, bond[1].index)
        push!(ids, bond[2].index)
    end
    return ids .+ 1
end

function onlychanges(a::Observable)
    o = Observable(a[])

    on(a) do a
        if a != o[]
            o[] = a
        end
    end
    return o
end
onlychanges(x) = x

observe(x) = x isa Observable ? x : Observable(x)

function visgradients(iso::Iso, x=coords(iso.data))
    dx = mapreduce(hcat, eachcol(x)) do c
        ISOKANN.dchidx(iso, c)
    end

    plotmol(x, iso.data.sim.pysim, grad=dx, showatoms=false)
end

plotmol(iso::Iso; kwargs...) = plotmol(Observable(iso); kwargs...)

function plotmol(o::Observable{<:Iso}; kwargs...)
    iso = o[]
    c = Observable(coords(iso.data))
    on(o) do val
        c[] = coords(iso.data)
    end
    plotmol(c, iso.data.sim.pysim; kwargs...)
end

function plot_pairs!(ax, c, pairs; kwargs...)
    c = @lift reshape($c, 3, :)
    inds = collect(Iterators.flatten(pairs))
    b = @lift map(Point3, eachcol($c[:, inds]))
    linesegments!(ax, b; kwargs...)
end

function plot_mi!(ax, c, iso, exponent=8)
    pairs = ISOKANN.featurepairs(iso.data)
    mi = ISOKANN.mutual_information(iso)
    mi = (mi ./ maximum(mi)).^ exponent
    color = [(:blue, m) for m in mi]
    plot_pairs!(ax, c, pairs; color, transparency=true)
    ax
end

mutable struct VisualizationObject
    coords
    fig
    ax
    plots
    iso
    meta
end

function VisualizationObject(iso::Iso)
    fig, coords = plotmol(iso, showbonds=false)

    VisualizationObject(coords, fig, fig.content[2], nothing, iso, Dict())
end

function plot_mi!(v::VisualizationObject)
    plot_mi!(v.ax, v.coords, v.iso)
end

function plotmol(c, pysim, color=1; grad=nothing, kwargs...)
    c = observe(c)
    color = observe(color)

    fig = Figure(size=(1280, 960))
    
    frameselector = SliderGrid(fig[1, 1],
        (label="Frame", range=@lift(1:size($c, 2)), startvalue=1))

    on(c) do c
        n = size(c, 2)
        frameselector.sliders[1].value[] = n
    end

    i = frameselector.sliders[1].value
    col = @lift $c[:, $i]


    ax = LScene(fig[2, 1], show_axis=false)
    plotmol!(ax, col, pysim, color; kwargs...)

    if !isnothing(grad)
        grad = @lift($(observe(grad))[:, $i])
        plotgrad!(ax, @lift(reshape($col, 3, :)), @lift(reshape($grad, 3, :)),
            arrowsize=0.01, lengthscale=0.2, linecolor=:red, linewidth=0.001)
    end

    return (;fig, coords=col)
end

function plotgrad!(ax, c::Observable{<:AbstractMatrix}, dc::Observable{<:AbstractMatrix}; kwargs...)

    x = @lift vec($c[1, :])
    y = @lift vec($c[2, :])
    z = @lift vec($c[3, :])
    u = @lift vec($dc[1, :])
    v = @lift vec($dc[2, :])
    w = @lift vec($dc[3, :])

    @show x[] |> size
    @show u[] |> size

    arrows!(ax, x, y, z, u, v, w; kwargs...)
end

function plotmol!(ax, c, pysim, color; showbonds=true, showatoms=false, showbackbone=true, alpha=1.0, linewidth=4)
    z = zeros(3, 0)

    c = @lift if length($c) > 0
        reshape($c, 3, :)
    else
        z
    end

    a = @lift if $showatoms
        $c
    else
        z
    end

    @show maxatoms = 4767 #size(c,2)

    cainds = ISOKANN.OpenMM.calpha_inds(pysim)
    cainds = filter(<(maxatoms), cainds)
    p = @lift if $showbackbone
        $c[:, cainds]
    else
        z
    end

    ids = bondids(pysim)
    ids = reduce(vcat, 
        filter(x -> all(x .< maxatoms),
            eachcol(reshape(ids, 2, :))
        ))

    b = @lift if $showbonds
        $c[:, ids]
    else
        z
    end

    color = onlychanges(color)

    meshscatter!(ax, onlychanges(a), markersize=0.1, color=@lift($color .* ones(size($a, 2))), colorrange=(0.0, 1.0), colormap=:roma,)
    lines!(ax, onlychanges(p); linewidth, color=color, colorrange=(0.0, 1.0), colormap=:roma,)
    linesegments!(ax, onlychanges(b); color=color, colorrange=(0.0, 1.0), colormap=:roma, alpha)


    ax
end

function plotmol!(ax, iso::Iso, i; kwargs...)
    coords = @lift ISOKANN.align(iso.data.coords[1][:, $i], iso.data.coords[1][:, 1])
    color = @lift(iso.model(iso.data.features[1][:, $i]) |> cpu)
    plotmol!(ax, coords, iso.data.sim.pysim, color; kwargs...)
end



function dashboard(iso::Iso, session=nothing)
    coords = Observable(iso.data.coords[1] |> cpu)
    chis = Observable(ISOKANN.chis(iso) |> vec |> cpu)
    icur = Observable(1)
    losses = Observable(iso.losses |> cpu)

    imin = @lift argmin($chis)
    imax = @lift argmax($chis)

    n = @lift size($coords, 2)

    fig = Figure(size=(1920, 1000))


    showbackbone = Toggle(fig, active=true)
    showatoms = Toggle(fig)
    showbonds = Toggle(fig, active=true)
    showextrema = Toggle(fig, active=true)

    labels = WGLMakie.Label.(Ref(fig), ["Backbone", "Bonds", "Atoms", "Extrema"])

    fig[1, 2] = grid!(hcat(labels, [showbackbone, showbonds, showatoms, showextrema]))

    colsize!(fig.layout, 1, Relative(2 / 5))
    colsize!(fig.layout, 2, Relative(3 / 5))


    run = Toggle(fig)
    contsampling = Toggle(fig)
    uniformsampling = Toggle(fig)
    adaptivesampling = Toggle(fig)

    fig[1, 1] = grid!(hcat(WGLMakie.Label.(Ref(fig), ["Run", "Trajectory", "Uniform", "Fill-In"]), [run, contsampling, uniformsampling, adaptivesampling]))


    ax = LScene(fig[1:6, 2], show_axis=false)


    plotmol!(ax, iso, icur, linewidth=6; showbackbone=showbackbone.active, showatoms=showatoms.active, showbonds=showbonds.active)

    let showbackbone = @lift($(showextrema.active) && $(showbackbone.active)),
        showatoms = @lift($(showextrema.active) && $(showatoms.active)),
        showbonds = @lift($(showextrema.active) && $(showbonds.active))

        plotmol!(ax, iso, imax; showbackbone, showatoms, showbonds)
        plotmol!(ax, iso, imin; showbackbone, showatoms, showbonds)
    end


    lines(fig[2, 1], losses, axis=(yscale=log10, limits=@lift((((1, max(length($losses), 2)), nothing)))))
    WGLMakie.scatter(fig[3, 1], chis, axis=(limits=@lift((1, $n, 0, 1)),))

    frameselector = SliderGrid(fig[4, 1],
        (label="Data Frame", range=@lift(1:$n), startvalue=n)
    )

    connect!(icur, frameselector.sliders[1].value)


    on(run.active) do e
        e || return

        global ISRUNNING
        if ISRUNNING
            run.active[] = false
            return
        else
            ISRUNNING = true
        end

        ThreadPools.@tspawnat 1 begin
            try
                last = time()
                while (isnothing(session) || isready(session)) && run.active[]

                    run!(iso)
                    adaptivesampling.active[] && ISOKANN.resample_kde!(iso, 1; padding=0.01, bandwidth=0.1)
                    uniformsampling.active[] && ISOKANN.adddata!(iso, 1, keepedges=false)
                    contsampling.active[] && continuedata!(iso)

                    cutoff = 1000
                    if length(iso.data) > cutoff
                        iso.data = iso.data[end-cutoff+1:end]
                    end



                    if time() - last > 0.1
                        chis[] = ISOKANN.chis(iso) |> vec |> cpu
                        coords[] = iso.data.coords[1]
                        icur[] = n[]
                        losses[] = iso.losses
                        last = time()
                    end
                end
            finally
                ISRUNNING = false
            end
        end
    end


    run_react = WGLMakie.Makie.Button(fig[5, 1], label="Compute reactive path")
    reactpath = Observable([1])
    react_select = SliderGrid(fig[6, 1],
        (label="Reactive Frame", range=@lift(1:max(1, length($reactpath))), startvalue=1),
        (label="Sigma", range=0.01:0.01:1, startvalue=0.1)
    )
    connect!(icur, @lift(reactpath[][$(react_select.sliders[1].value)]))

    on(run_react.clicks) do e
        ids = reactive_path(iso; sigma=react_select.sliders[2].value[], maxjump=1)
        reactpath[] = ids
    end

    fig
end

function continuedata!(iso)
    iso.data = ISOKANN.addcoords(iso.data, reshape(iso.data.coords[2][:, 1, end], :, 1))
end




function livevis(iso::Iso)
    coords = iso.data.coords[1][:, 1]
    a = Observable(coords)
    b = Observable(coords)
    o = Observable(coords)
    data = Observable(chis(iso) |> vec)

    N = @lift length($data)

    losses = Observable(iso.losses)

    col = Observable(repeat([0.0], div(length(coords), 3)))

    align_to_prev(x) = ISOKANN.align(x, a[])

    function update!(; kwargs...)
        #@views newcoords = reshape(iso.data.coords[1], 3, :)[:, nold+1:end]
        #nold += size(newcoords, 2)
        #scatter!(newcoords, markersize=0.01, color=(:blue, 0.2))


        c = chis(iso) |> vec |> cpu
        data[] = c
        i = argmin(c)
        a[] = iso.data.coords[1][:, i] #|> align_to_prev
        i = argmax(c)
        b[] = iso.data.coords[1][:, i] #|> align_to_prev
        o[] = iso.data.coords[1][:, end] #|> align_to_prev

        losses[] .= iso.losses
        notify(losses)

        col[] .= c[end]
        notify(col)

        #allcoords[] = reshape(iso.data.coords[1], 3, :)
        #colors[] = repeat(iso.model(iso.data.features[1]) |> vec, inner=22)
    end
    update!()

    fig = Figure(size=(1000, 1000))
    ax = LScene(fig[1:2, 1], show_axis=false)
    let pysim = iso.data.sim.pysim
        plotmol!(ax, o, pysim, col)
        plotmol!(ax, a, pysim, 0.0)
        plotmol!(ax, b, pysim, 1.0)
    end

    lines(fig[3, 1], losses, axis=(yscale=log10, limits=@lift(((1, length($losses)), nothing))))

    scatter(fig[4, 1][1, 1], data, axis=(limits=(nothing, nothing),))

    frameselector = SliderGrid(fig[4, 1][2, 1],
        (label="Frame", range=@lift(1:$N), startvalue=1)
    )

    on(frameselector.sliders[1].value) do i
        o[] = iso.data.coords[1][:, i]
        col[] .= data[][i]
        notify(col)
    end

    #=
    on(events(fig).mousebutton) do event

        a, i = pick(fig)
        i > 0 || return
        @show i
        x = iso.data.coords[1][:, i] # |> align_to_prev
        o[] = x
    end
    =#

    #allcoords = Observable(reshape(iso.data.coords[1], 3, :))
    #colors = Observable(repeat(iso.model(iso.data.features[1]) |> vec, inner=22))
    #scatter!(allcoords, markersize=0.1, color = colors, colormap=:thermal)
    #nold = 0


    return fig, update!
end

function scatterdata!(fig, iso::Iso, data)
    scatter!(fig, data)
end

function addvis!(iso)
    plt, up = livevis(iso)
    iso.loggers = [up]
    plt
end

function showmakie(iso=Iso(OpenMMSimulation(steps=20), loggers=[]))
    run!(iso, 2)
    plt, up = livevis(iso)
    display(plt)
    function run(n)
        for i in 1:n
            run!(iso)
            ISOKANN.resample_kde!(iso, 1; padding=0)
            up()
        end
    end

    return plt, run, iso
end

function voxelize(data, n)
    data = reshape(data, 3, :)
    e = extrema(data, dims=2)
    e = zip(e...) |> collect
    magn = e[2] .- e[1]
    d = data .- e[1]
    v = round.(Int, d ./ magn .* (n - 1)) .+ 1
    x = zeros(Int, n, n, n)
    for i in eachcol(v)
        x[CartesianIndex(Tuple(i))] += 1
    end
    return x ./ maximum(x)
end

function scatter_rc(rc, chi; markersize=10, msselect=30, kwargs...)
    fig = Figure()
    ax = Axis3(fig[1, 1])
    ms = lift(rc) do rc
        ones(size(rc, 2)) .* markersize
    end
    fig, ax, plt =
    #plt= 
        WGLMakie.scatter(rc, color=chi;
            colormap=:thermal,
            markersize=ms,
            strokewidth=0, 
            kwargs...
        )
   

    i = Observable(1)

    function hover(inspector, plt, index)
        i[] = index
        return true
    end
    d = DataInspector(ax.scene)
    plt.inspector_hover = hover

    #=
    on(events(fig).mouseposition) do event
        mb = events(fig).mousebutton[]
        #if mb.button == Mouse.left# && event.action == Mouse.press
            plt, ind = pick(fig,100)
            @show ind
            if ind > 0
                i[] = ind
            end
        #end
    end
    =#

    on(i) do ind
        if i[] in eachindex(ms[]) 
            ms[] .= markersize
        end
        if ind in eachindex(ms[])
            ms[][ind] = msselect
            notify(ms)
        end
    end


    #Makie.Camera3D(ax.scene, projectiontype=Makie.Orthographic)
    fig, ax, plt, i
end


""" visualize(iso, reactioncoords, chi)

Scatter plot of the reaction coordinates `reactioncoords` with `chi` as color.
Clicking on any of the data points will visualize the corresponding molecular structure in a separate figure.
"""
function visualize(iso, reactioncoords, chi)
    fig, ax, plt, i = scatter_rc(reactioncoords, chi)
    c = lift(i) do i
        if i > 0
            ISOKANN.centercoords(iso.data.coords[1][:, i])
        else
            zeros(size(iso.data.coords[1], 1))
        end
    end
    fig2 = ISOKANN.plotmol(c, iso.data.sim.pysim, showbonds=false)
    App() do s
        DOM.div([fig, fig2])
    end
end


function visualize(iso::Observable, rcmap)

    #WGLMakie.activate!(resize_to=:parent)
    xs = lift(coords, iso) |> onlychanges
    reactioncoords = lift(rcmap, xs) |> onlychanges
    chi = lift(iso) do iso
        chis(iso) |> vec |> cpu
    end |> onlychanges

    pysim = iso[].data.sim.pysim

    fig, ax, plt, i = scatter_rc(reactioncoords, chi)

    on(xs) do xs
        i[] = size(xs,2)
    end

    selcoords = lift(i) do i
        if i > 0
            #ISOKANN.centercoords(xs[][:, i])
            ISOKANN.align(xs[][:,i], xs[][:,1])
        else
            zeros(size(xs[], 1))
        end
    end

    #fig2 = ISOKANN.plotmol(selcoords, pysim, showbonds=false)
    fig2 = Figure()
    ax = LScene(fig2[1,1], show_axis=false)
    plotmol!(ax, selcoords, pysim, observe(1),showatoms=false)

    App() do session
        #fig
        DOM.div([fig, fig2])
    end
end

struct IsoVis
    iso
    obs
    fig
end

function IsoVis(iso, rcmap=autorcmap(iso)) 
    obs = observe(iso)
    app = visualize(obs, rcmap)
    @show server = Bonito.get_server()
    route!(server, "/" => app)
    #display(app)
    IsoVis(iso, obs, app)
end

function autorcmap(iso)
    dim = OpenMM.dim(iso.data.sim)
    
    if dim == 66
        rc(x)=vcat(ISOKANN.phi(x)', ISOKANN.psi(x)')
    elseif dim == 1773
        rc = map([3:10, 14:17, 22:31]) do inds
            ISOKANN.ca_rmsd(inds, "data/villin/1yrf.pdb", "data/villin nowater.pdb")
        end
    end
    return rc
end

function run!(v::IsoVis; generations=1, iter=1, kde=0, update=0.1)
    t = 0.
    for i in 1:generations
        t += @elapsed runadaptive!(v.iso, generations=1; kde, iter)
        if t > update
            @show t
            t = 0.
            @time notify(v.obs)
        end
    end
    notify(v.obs)
end

end