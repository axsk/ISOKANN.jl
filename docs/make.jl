using Documenter
using InteractiveUtils: @time_imports
@time @time_imports using ISOKANN

makedocs(
    sitename="ISOKANN",
    format=Documenter.HTML(),
    modules=[ISOKANN],
    warnonly=true
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo="github.com/axsk/ISOKANN.jl.git"
)
