function test()
    test_equivariance()
    test_grad()
end

function test_equivariance()
    layer = VecNeur(3,10,3,10)

    in = rand(3,10)

    rot = [0 1 0
          -1 0 0
           0 0 1]

    @assert rot*layer(in) == layer(rot*in)
end

function test_grad()
    in = rand(3,10)
    out = rand(3,10)
    layes = VecNeur(3,10,3,10)

    Zygote.withgradient(layer) do layer
        sum(abs2, layer(in) - out)
    end
end
