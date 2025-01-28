using ISOKANN, JLD2, Plots, Dates
using ISOKANN: coords  # since Plots also exports this 
includet("../src/utils/picking.jl")

test() = main(10000)

function main(stride=100, iter0=3000, iter=1000)
    global iso
    init_rmsds()

    xs = jldopen("data/villin/villin superlong.jld2")["xs"][:, 1:stride:200_000]
    xs = picking_aligned(xs, 1000)[1]

    sim = OpenMMSimulation(pdb="data/villin nowater.pdb", steps=5000)
    featurizer = ISOKANN.OpenMM.FeaturesPairs(sim, maxfeatures=2000, maxdist=Inf)

    @time "simulating initial data" data = SimulationData(sim, xs, 2; featurizer)

    iso = Iso(data)

    @time "initial training" run!(iso, iter0)

    date = Dates.now()

    for i in 0:iter
        p = myplot(iso)
        n = size(iso.data.coords[1], 2)
        savefig("out/vil-$date-$n.png")
        #i % 10 == 0 && savefig("out/vill$i.png")
        runadaptive!(iso; generations=10, kde=1, iter=20)
    end

    return iso
end

function iso_implicit(ndata=2000)
    global iso
    global dir
    dir = mkdir("out/vil-$(Dates.now())")

    xs = JLD2.load("vil starts.jld2", "xs")
    xs, q, d, res = picking_aligned(xs, ndata)

    sim = OpenMMSimulation(
        pdb="data/villin nowater.pdb",
        steps=5000,
        forcefields=OpenMM.FORCE_AMBER_IMPLICIT)

    featurizer = ISOKANN.OpenMM.FeaturesPairs(sim,
        maxfeatures=2000,
        maxdist=Inf)

    @time "simulating initial data" data = SimulationData(sim, xs, 2; featurizer)

    iso = Iso(data)

    @time "initial training" run!(iso, 1000)
    return iso
end

function trainloop(iso, n)
    for i in 1:n
        runadaptive!(iso; generations=10, kde=1, iter=20)
        p = myplot(iso)
        n = size(iso.data.coords[1], 2)
        savefig("$(dir)/vil-$n.png")
        @time if i % 10 == 0
            JLD2.save("$(dir)/iso.jld2", "iso", iso)
            JLD2.save("$(dir)/isocpu.jld2", "iso", cpu(iso))
            ISOKANN.save("$(dir)/iso.iso", iso)
            wglscatter(iso)
        end
    end
end

function myplot(iso)
    p = plot_training(iso)
    s = scatter_rmsd(iso)

    plot(plot(p.subplots[1]), plot(p.subplots[2]), s, layout=(3, 1), size=(500, 900))
end

function init_rmsds(pdb="data/villin nowater.pdb", pdbref="data/villin/1yrf.pdb")
    global ca, car, xr

    ca = OpenMM.calpha_inds(OpenMMSimulation(pdb=pdb))
    refstruct = OpenMMSimulation(pdb=pdbref)
    car = OpenMM.calpha_inds(refstruct)
    xr = coords(refstruct)
end
#=
function rsmd_cas(x::AbstractVector, inds)
    x1 = reshape(x, 3, :)[:, ca[inds]]
    x2 = reshape(xr, 3, :)[:, car[inds]]
    ISOKANN.aligned_rmsd(x1, x2)
end

rsmd_coords(x::Matrix) = [rsmd_cas(col, ind) for ind in [2:11, 13:20, 20:31], col in eachcol(x)]
=#

# inds from jakob [2:11, 13:20, 20:31]
# my inds from looking at the pdb and leaving one slack at the ends


function scatter_rmsd(x; rmsds=rmsds, kwargs...)

    c = rmsds(x)
    scatter3d(c[1, :], c[2, :], c[3, :],
        xlabel="A", ylabel="B", zlabel="C",
        title="RSMDs", label="",
        lims=(0, :auto); kwargs...)
end

scatter_rmsd(iso::Iso; kwargs...) =
    scatter_rmsd(coords(iso.data),
        marker_z=chis(iso) |> cpu |> vec,
        ms=(1:length(iso.data)) .* (5 / length(iso.data)),
        ; kwargs...
    )

#=
struct ReactionCoordsRMSD
    inds
    refcoords
end

function CA_RMSD(cainds, pdb="data/villin nowater.pdb", pdbref="data/villin/1yrf.pdb")

    ca = OpenMM.calpha_inds(OpenMMSimulation(pdb=pdb))
    inds = ca[cainds]

    refstruct = OpenMMSimulation(pdb=pdbref)
    car = OpenMM.calpha_inds(refstruct)
    xr = coords(refstruct)
    refcoords = reshape(xr, 3, :)[:, car[cainds]]

    ReactionCoordsRMSD(inds, refcoords)
end
=#

rmsds = map([3:10, 14:17, 22:31]) do inds
    ISOKANN.ca_rmsd(inds, "data/villin/1yrf.pdb","data/villin nowater.pdb")
end
#=
function (r::ReactionCoordsRMSD)(x::AbstractVector)
    x = reshape(x, 3, :)[:, r.inds]
    return ISOKANN.aligned_rmsd(x, r.refcoords)
end

(r::ReactionCoordsRMSD)(xs::AbstractMatrix) = map(r, eachcol(xs))
(rs::Vector{ReactionCoordsRMSD})(xs::AbstractMatrix) = [r(col) for r in rs, col in eachcol(xs)]
=#
function plot_picking(xs, n=100)

    kwargs = (; xlabel="A", ylabel="B", zlabel="C", title="RSMDs", label="", lims=(0, :auto))
    p1 = scatter3d(eachrow(rmsds(xs))...; kwargs..., title="RMSDs")
    p2 = scatter3d(eachrow(rmsds(picking_aligned(xs, n)[1]))...; kwargs..., title="RMSDs of picked")
    p3 = scatter3d(eachrow(picking(rmsds(xs), n)[1])...; kwargs..., title="picked RMSDs")
    plot(p1, p2, p3, layout=(1, 3), size=(1200, 400))
end

using ISOKANN: WGLMakie
using WGLMakie: Bonito
using Bonito

WGLMakie.activate!(resize_to=:parent)



function wgl_rmsd(iso)
    x = coords(iso.data)
    c = rmsds(x)
    chi = chis(iso) |> cpu |> vec

    WGLMakie.scatter(c, color=chi, colormap=:thermal,)
end

function inspect(plot)
    function hover(inspector, plt, index, hovered_child)
        @show plt, index, hovered_child
    end

    plot.inspector_hover = hover
end

function wglscatter(iso; path="$dir/scatter.html")
    plot = wgl_rmsd(iso)
    wgl_savefig(path, plot)
    println(path)
end

function wgl_savefig(path, plot)
    open(path, "w") do io
        println(
            io,
            """
<html>
    <head>
    </head>
    <body>
"""
        )
        Page(exportable=true, offline=true)
        # Then, you can just inline plots or whatever you want :)
        # Of course it would make more sense to put this into a single app
        app = App() do
            #C(x;kw...) = Card(x; height="fit-content", width="fit-content", kw...)
            #figure = (; size=(300, 300))
            #f1 = scatter(1:4; figure)
            #f2 = mesh(load(assetpath("brain.stl")); figure)
            #C(DOM.div(
            #    Bonito.StylableSlider(1:100),
            #    Row(C(f1), C(f2))
            #); padding="30px", margin="15px")
            plot
        end
        show(io, MIME"text/html"(), app)
        # or anything else from Bonito, or that can be displayed as html:
        println(
            io,
            """
    </body>
</html>
"""
        )
    end
end

println("script loaded, to run it use `main()`")