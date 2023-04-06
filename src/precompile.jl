using SnoopPrecompile

@precompile_all_calls begin
    run!(IsoRun(
        nd = 1, np = 1, nl = 1,
        ny = 2, nk = 1,
        loggers=[]
    ))
end
