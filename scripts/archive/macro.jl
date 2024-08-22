using MacroTools: splitdef, splitarg, combinedef

argnames(args) = first.(splitarg.(args))


# TODO: this does not work with splatting yet

macro autokw(expr)
  d = splitdef(expr)

  kws = vcat(d[:args], d[:kwargs])

  rets = (d[:body].args[end])
  a1 = argnames(d[:args])
  a2 = argnames(d[:kwargs])
  retkw = unique(vcat(a1, a2, rets))

  d[:args] = []
  d[:kwargs] = kws
  d[:body] =
    quote
      $(rets) = $(d[:name])($(a1...); $(a2...))
      return (; $(retkw...))
    end

  newdef = combinedef(d)

  quote
    $(esc(expr))
    $(esc(newdef))
  end


end

