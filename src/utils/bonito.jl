using Bonito
using WGLMakie

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
