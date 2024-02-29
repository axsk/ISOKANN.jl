using Flux
using Optimisers
using Zygote

model = Chain(
  Dense(5500 => 300, relu),                # 1_650_300 parameters
  Dense(300 => 100, relu),                 # 30_100 parameters
  Dense(100 => 10, relu),                  # 1_010 parameters
  Dense(10 => 1, relu),                    # 11 parameters
  Dense(1 => 1),                        # 2 parameters
)                   # Total: 10 arrays, 1_681_423 parameters, 6.415 MiB.

opt = Optimisers.setup(Optimisers.OptimiserChain(Optimisers.WeightDecay(), Optimisers.Adam()), model)

function train_step!(model, xs::AbstractMatrix, target::AbstractMatrix, opt)
  l, grad = let xs = xs  # `let` allows xs to not be boxed
    Zygote.withgradient(model) do model
      sum(abs2, model(xs) .- target) / size(target, 2)
    end
  end
  Optimisers.update!(opt, model, grad[1])
  return l
end

@benchmark for i in 1:1
  train_step!(model, rand(5500, 128), rand(1, 128), opt)
end
