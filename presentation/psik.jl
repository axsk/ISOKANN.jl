include("../src/forced/IsoForce.jl")

import Optimisers

# adaptive sampling + optimal control on doublewell

IsoForce.isokann(usecontrol=false)
IsoForce.isokann(usecontrol=true)

# variance control even for nk=1
IsoForce.isokann(usecontrol=true, nkoop=1, opt=Optimisers.Adam(0.001))


# alanine dipeptide
# mollySDE failing

# visualiztion of reaction path
# mu opiod

# visualiztion of reaction path
