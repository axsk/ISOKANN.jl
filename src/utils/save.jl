
import FileIO

FileIO.add_loader(format"PDB", :ISOKANN)
FileIO.add_saver(format"PDB", :ISOKANN)

function fileio_load(f::File{format"PDB"}; lib=:mdtraj)
    if lib == :chemfiles
        readchemfile(f.filename)
    elseif lib == :mdtraj
        load_trajectory(f.filename)
    else
        error()
    end
end



function fileio_save(f::File{format"PDB"}, data, top; lib=:mdtraj)
    if lib == :mdtraj
        save_trajectory(f.filename, data; top)
    elseif lib == :chemfiles
        writechemfile(f.filename, data; source=top)
    else
        error()
    end
end

function test_loadsave(f::String, coords, sim::OpenMMSimulation)
    pdb = ISOKANN.pdbfile(sim)
    writers = Dict(
        "wpytraj" => () -> ISOKANN.save_trajectory(f, coords, top=pdb),
        "wchemf " => () -> ISOKANN.writechemfile(f, coords, source=pdb),
        "wopenmm" => () -> OpenMM.savecoords(f, sim, coords),)

    readers = Dict(
        "rpytraj" => () -> ISOKANN.load_trajectory(f),
        "rchemf " => () -> ISOKANN.readchemfile(f))

    for (n, w) in writers
        @time n w()
        for (n2, r) in readers
            #try
            x = @time n * " " * n2 r()
            #return x, coords
            @show x - coords |> norm
            @assert isapprox(x, coords, atol=1e-2)
            # catch


            #end
        end
    end
end
