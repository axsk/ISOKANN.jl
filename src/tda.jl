using StatsBase: std, mean
using KernelFunctions: SqExponentialKernel
using Graphs: SimpleGraph, connected_components
using WGLMakie, GraphMakie

## Kernel two sample test
# https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Kernel_two-sample_test

# precompute the self-kernel k(x_i, x_j)
function selfkernel(x::AbstractMatrix, kernel)
  kx = 0.0
  for i in axes(x, 2)
    for j in axes(x, 2)
      kx += @views kernel(x[:, i], x[:, j])
    end
  end
  kx /= size(x, 2)^2
  return kx
end

# batched precomputation of self-kernel for multiple densities
function selfkernel(x::AbstractArray{<:Any,3}, kernel)
  KX = map(axes(x, 3)) do l
    @views selfkernel(x[:, :, l], kernel)
  end
  return KX
end

# twosampletest comparing two sample clouds xs, ys. accepts precomputed self-kernels
function twosampletest(xs::AbstractMatrix, ys::AbstractMatrix, kernel;
  kx=selfkernel(xs, kernel),
  ky=selfkernel(xs, kernel))

  m = 0.0
  for i in axes(xs, 2)
    for j in axes(ys, 2)
      m -= 2 * @views kernel(xs[:, i], ys[:, j])
    end
  end
  m /= size(xs, 2) * size(ys, 2)
  m += kx + ky
  return m
end

# pairwise twosampletest for multiple densities
function pairwise_twosampletest(xs::AbstractArray{<:Any,3}, kernel; KX=selfkernel(xs, kernel))
  n = size(xs, 3)
  M = zeros(n, n)
  for l in 1:n
    for m in 1:n
      M[l, m] = @views twosampletest(xs[:, :, l], xs[:, :, m], kernel, kx=KX[l], ky=KX[m])
    end
  end
  return M
end



## Topological Data Analysis

function tda(ys, chi; kernel=SqExponentialKernel(), windows=10, overlap=0.5, normalize=true)

  # normalize input data: TODO: some work is required here, in accordance with the kernel choice
  if normalize
    ys ./= mean(std(ys, dims=2), dims=3)
    ys ./= sqrt(size(ys, 1))
  end

  r = range(start=0, stop=1, length=windows + 1)

  # find indices of samples in each window
  ixs = []
  for i in 1:windows
    push!(ixs, findall(c -> r[i] <= c < r[i+1], chi))
  end

  # pre-compute the self-overlap / selfkernel
  KX = selfkernel(ys, kernel)
  @show KX

  if overlap == :auto
    overlap = median(KX) * 2 / 3
    @show overlap
  end

  # compute the nodes for each level by finding the components connected by the twosampletest
  levelnodes = Vector{Vector{Int}}[]
  for ix in ixs
    A = pairwise_twosampletest(ys[:, :, ix], kernel) .< overlap
    components = connected_components(SimpleGraph(A))
    nodes = [ix[comp] for comp in components] # map in-layer indices back go global inds
    push!(levelnodes, nodes)
  end

  # flatten list of all nodes
  nodes = reduce(vcat, levelnodes)

  nodeindex = zeros(Int, size(ys, 3))
  for (i, n) in enumerate(nodes)
    nodeindex[n] .= i
  end

  # create a graph with edges between nodes that have at least two points connected by the twosampletest
  A = zeros(Bool, length(nodes), length(nodes))
  for i in 1:windows-1
    for n in levelnodes[i]
      @show n
      for m in levelnodes[i+1]
        for i in n, j in m
          if @views twosampletest(ys[:, :, i], ys[:, :, j], kernel) < overlap
            let a = nodeindex[i], b = nodeindex[j]
              A[a, b] = A[b, a] = 1
            end
            break
          end
        end
      end
    end
  end

  return nodes, A
end

function test(overlap=0.1)
  ys = rand(10, 10, 100)
  chi = rand(100)
  nodes, A = tda(ys, chi; overlap)
  c = map(nodes) do n
    chi[n] |> mean
  end
  graphplot(SimpleGraph(A), nodes, node_color=c)
end