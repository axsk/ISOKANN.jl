using WGLMakie
using ISOKANN

function bondids(pysim)
    ids = Int[]
    for bond in pysim.topology.bonds()
        push!(ids, bond[1].index)
        push!(ids, bond[2].index)
    end
    return ids .+ 1
end

function plotmol!(ax, c, pysim, color; showbonds=true, showatoms=true, showbackbone=true, alpha=1.0, linewidth=4)
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

    cainds = ISOKANN.OpenMM.calpha_inds(pysim)
    p = @lift if $showbackbone
        $c[:, cainds]
    else
        z
    end

    ids = bondids(pysim)

    b = @lift if $showbonds
        $c[:, ids]
    else
        z
    end


    #if atoms
    meshscatter!(ax, a, markersize=0.008, color=@lift($color .* ones(size($a, 2))),
        colorrange=(0.0, 1.0), colormap=:roma,)
    #end

    #if cas
    #ca = @lift
    lines!(ax, p; linewidth, color=color, colorrange=(0.0, 1.0), colormap=:roma,)
    #end

    #if bonds

    #bonds = @lift 
    linesegments!(ax, b; color=color, colorrange=(0.0, 1.0), colormap=:roma, alpha)
    #end
    ax
end

function plotmol!(ax, iso::Iso2, i; kwargs...)
    coords = @lift ISOKANN.align(iso.data.coords[1][:, $i], iso.data.coords[1][:, 1])
    color = @lift(iso.model(iso.data.features[1][:, $i]))
    plotmol!(ax, coords, iso.data.sim.pysim, color; kwargs...)
end

#dashboard(iso::Iso2) = dashboard(Observable(Iso2))
function dashboard(iso::Iso2)
    coords = Observable(iso.data.coords[1])
    chis = Observable(ISOKANN.chis(iso) |> vec |> cpu)
    icur = Observable(1)
    losses = Observable(iso.losses)

    imin = @lift argmin($chis)
    imax = @lift argmax($chis)

    n = @lift size($coords, 2)

    fig = Figure(size=(1600, 1000))


    displaygrid = fig[1, 3] = GridLayout()
    showbackbone = Toggle(displaygrid[1, 1], active=true).active
    showatoms = Toggle(displaygrid[2, 1]).active
    showbonds = Toggle(displaygrid[3, 1], active=true).active
    showextrema = Toggle(displaygrid[4, 1], active=true).active

    ax = LScene(fig[1:6, 2:3], show_axis=false)


    plotmol!(ax, iso, icur, linewidth=6; showbackbone, showatoms, showbonds)

    let showbackbone = @lift($showextrema && $showbackbone),
        showatoms = @lift($showextrema && $showatoms),
        showbonds = @lift($showextrema && $showbonds)

        plotmol!(ax, iso, imax; showbackbone, showatoms, showbonds)
        plotmol!(ax, iso, imin; showbackbone, showatoms, showbonds)
    end


    lines(fig[3, 1], losses, axis=(yscale=log10, limits=@lift((((1, max(length($losses), 2)), nothing)))))
    scatter(fig[4, 1][1, 1], chis, axis=(limits=@lift((1, $n, 0, 1)),))

    frameselector = SliderGrid(fig[5, 1],
        (label="Data Frame", range=@lift(1:$n), startvalue=n)
    )

    connect!(icur, frameselector.sliders[1].value)

    run = Toggle(fig[1:2, 1][1, 2])

    adaptivesampling = Toggle(fig[1:2, 1][2, 2])

    uniformsampling = Toggle(fig[1:2, 1][3, 2])

    contsampling = Toggle(fig[1:2, 1][4, 2])



    on(run.active) do e
        @show e
        Threads.@spawn begin
            for _ in 1:1000
                run.active[] || break
                run!(iso)
                adaptivesampling.active[] && ISOKANN.resample_kde!(iso, 1; padding=0.0)
                uniformsampling.active[] && ISOKANN.adddata!(iso, 1, keepedges=false)
                contsampling.active[] && continuedata!(iso)

                chis[] = ISOKANN.chis(iso) |> vec |> cpu
                coords[] = iso.data.coords[1]
                icur[] = n[]
                losses[] = iso.losses
                sleep(0.001)
            end
        end
    end


    run_react = WGLMakie.Makie.Button(fig, label="Compute reactive path")
    reactpath = Observable([1])
    react_select = SliderGrid(fig[7, 1],
        (label="Reactive Frame", range=@lift(1:max(1, length($reactpath))), startvalue=1),
        (label="Sigma", range=0.01:0.01:1, startvalue=0.1)
    )
    connect!(icur, @lift(reactpath[][$(react_select.sliders[1].value)]))

    on(run_react.clicks) do e
        ids, _ = reactive_path(iso; sigma=react_select.sliders[2].value[], maxjump=1)
        reactpath[] = ids
    end

    fig
end

function continuedata!(iso)
    iso.data = ISOKANN.addcoords(iso.data, reshape(iso.data.coords[2][:, 1, end], :, 1))
end




function livevis(iso::Iso2)
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

    on(events(fig).mousebutton) do event

        a, i = pick(fig)
        i > 0 || return
        @show i
        x = iso.data.coords[1][:, i] # |> align_to_prev
        o[] = x
    end

    #allcoords = Observable(reshape(iso.data.coords[1], 3, :))
    #colors = Observable(repeat(iso.model(iso.data.features[1]) |> vec, inner=22))
    #scatter!(allcoords, markersize=0.1, color = colors, colormap=:thermal)
    #nold = 0


    return fig, update!
end

function scatterdata!(fig, iso::Iso2, data)
    scatter!(fig, data)
end

function addvis!(iso)
    plt, up = livevis(iso)
    iso.loggers = [up]
    plt
end

function showmakie(iso=Iso2(OpenMMSimulation(steps=20), loggers=[]))
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