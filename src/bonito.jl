using Revise
using Bonito
using WGLMakie
using ISOKANN

includet("makie.jl")

#iso = Iso2(OpenMMSimulation(steps=30, pdb="data/vgv.pdb", forcefields=ISOKANN.OpenMM.FORCE_AMBER_IMPLICIT), loggers=[], opt=ISOKANN.NesterovRegularized(1e-3, 0), nx=10, nk=1, minibatch=64)


function content()
    isoui, isoo = isocreator()
    Grid(isoui)
    Grid(isoui, @lift(dashboard($isoo)))
end

logrange(x1, x2, n) = round.((10^y for y in range(log10(x1), log10(x2), length=n)), sigdigits=1)

function isocreator()
    pdb = Dropdown(["data/alanine-dipeptide-nowater.pdb", "data/vgv.pdb", "data/villin nowater.pdb"])
    steps = StylableSlider(1:100, value=10)
    temperature = StylableSlider(-10:70, value=30)
    optim = Dropdown(["Adam", "Nesterov"])
    learnrate = StylableSlider(sort([logrange(1e-4, 1e-2, 10); 1e-3]), value=1e-3)
    regularization = StylableSlider(sort([logrange(1e-6, 1e-3, 10); 1e-4]), value=1e-4)
    nx = StylableSlider(2:100, value=10)
    nk = StylableSlider(1:10, value=2)


    button = Button("Create")

    function create_iso()
        opt = if optim.value == "Adam"
            ISOKANN.AdamRegularized(learnrate.value[], regularization.value[])
        else
            ISOKANN.NesterovRegularized(learnrate.value[], regularization.value[])
        end

        iso = Iso2(
            OpenMMSimulation(
                steps=steps.value[],
                pdb=pdb.value[],
                forcefields=ISOKANN.OpenMM.FORCE_AMBER_IMPLICIT,
                temp=temperature.value[] + 272.15),
            loggers=[], opt=opt, nx=nx.value[], nk=nk.value[], minibatch=64, gpu=true)

        return iso
    end

    isoo = Observable(create_iso())

    on(button.value) do _
        isoo[] = create_iso()
    end

    return Card(Grid(
            pdb, nothing,
            steps, Bonito.Label(steps.value),
            temperature, Bonito.Label(temperature.value),
            learnrate, Bonito.Label(learnrate.value),
            regularization, Bonito.Label(regularization.value),
            button,
            columns="1fr min-content",
            justify_content="begin",
            align_items="center",
        ); width="300px",), isoo
end

app = App() do
    return content()
end