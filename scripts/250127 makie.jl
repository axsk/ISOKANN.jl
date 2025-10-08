using Bonito

Bonito.configure_server!(listen_port=3010, listen_url="localhost")

Bonito.get_server()

using WGLMakie

using ISOKANN



function scatter_rc(iso)
    x = coords(iso.data)
    rc = rmsds(x)
    chi = chis(iso) |> cpu |> vec
    scatter_rc(rc, chi)
end

function scatter_rc(rc, chi;
    markersize=5)
    fig = Figure()
    ax = Axis3(fig[1, 1])
    ms=Observable(ones(size(rc,2)) .* markersize)
    fig, ax, plt  =
    #plt= 
    WGLMakie.scatter(rc, color=chi;
        colormap=:thermal, 
        markersize=ms,
        strokewidth=0,
    )
    #d=DataInspector(ax.scene)

    i = Observable(0)

    function hover(inspector, plt, index)
        #@show plt, index
        i[] = index
        return true
    end

    #plt.inspector_hover = hover

    on(events(fig).mousebutton, priority =2 ) do event
        if event.button == Mouse.left && event.action == Mouse.press
                plt, ind = pick(fig)
                if ind > 0
                i[] > 0 && (ms[][i[]] = markersize)
                ms[][ind] = 20
                notify(ms)
                i[] = ind
                end
        end
    end
    

    Makie.Camera3D(ax.scene, projectiontype=Makie.Orthographic)
    fig, ax, plt, d, i
end

function serve(iso)
    fig,ax,plt,d,i=scatter_rc(iso)
    c = lift(i) do i
        if i > 0 
            ISOKANN.centercoords(iso.data.coords[1][:,i])
        else
            zeros(size(iso.data.coords[1],1))
        end
    end
    fig2 = ISOKANN.plotmol(c, iso.data.sim.pysim, showbonds=false)
    App() do s
       DOM.div([fig, fig2])
    end
end