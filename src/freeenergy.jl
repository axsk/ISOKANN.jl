"""
     marginal_free_energy(iso::Iso;nbins)

Compute the free energy from the density of chi values.

# Arguments
`iso` the Iso object.
`nbins` the number of bins of the histogram used for estimation.
# Returns
`F` the free energy energy surface of χ in kJ/mol up to an additive constant.
"""
function marginal_free_energy(iso::Iso;nbins=100)
    iso=(cpu(iso))
    # Parameters
    chivals = vec(chis(iso))
  
    sim = iso.data.sim
    kBT = 0.008314463 * OpenMM.temp(sim)
  
    # Create a histogram for the reaction coordinate values
    hist = fit(Histogram, chivals, nbins=nbins)
  
    # Bin centers
    edges = hist.edges[1]
    bin_centers = (edges[1:end-1] + edges[2:end]) ./ 2
  
    # Normalize histogram to get an estimate of the probability density.
    # Note: diff(edges) gives the bin widths.
    P = hist.weights ./ sum(hist.weights .* diff(edges))
  
    # Compute the free energy (up to an additive constant)
    F = -kBT * log.(P)
  
    # Shift free energy so that the minimum is zero (this is arbitrary)
    F .-= minimum(F)
  
    # Plot the free energy profile
    p=plot(bin_centers, F, xlabel="χ", ylabel="Free Energy [kJ/mol]",
        title="Free Energy Profile", legend=false,size=(600,600), frame=:box)
    display(p)
    return(F)
  end
  
  """
    constrained_free_energy(iso, xs; sim, steps)

Compute the free energy using Thermodynamic Integration. 
Starting from the levelset samples xs orhtogonal simulations estimate the 
mean force along χ, which is integrated to yield the PMF. 

# Arguments
`iso` the Iso object.
`xs`  the starting points (which should be well distributed in state space).
`sim` the simulation used for the orthongal sampling.
`steps` the number of steps in each orthogonal simulation.
# Returns
`F` the free energy energy surface of χ in kJ/mol up to an additive constant.
"""
  function constrained_free_energy(iso::Iso, xs; sim::OpenMMSimulation=iso.data.sim, steps=2000)
    iso = cpu(iso)
    n_states = size(xs, 2)
    mean_forces = zeros(n_states)
    mean_Z = zeros(n_states)
  
    dt   = OpenMM.stepsize(sim)
    gamma = OpenMM.friction(sim)
    kBT = 0.008314463 * OpenMM.temp(sim)
    m    = repeat(OpenMM.masses(sim), inner=3)
    chi_vals= [ iso.model(iso.data.featurizer(x))[1] for x in eachcol(xs)]
   
    for i in 1:n_states
        # Initialize state x for the i-th reaction coordinate point.
        x = xs[:, i]
        # Preallocate arrays
        lambdas = zeros(steps)
        Zs = zeros(steps)

        chi_level = chi_vals[i]
        println(chi_level)
  
        v = zeros(length(x))
        for j in 1:steps
            F = OpenMM.force(sim, x)
            dchi = ISOKANN.dchidx(iso, x)
            F_proj = dot(F, dchi) / dot(dchi, dchi)
            #Simulate on the orthogonal (it does not work to project the velocities)
            @. F -= F_proj * dchi
            db = randn(length(x))
            @. v += 1 / m * ((F - gamma * v) * dt + sqrt(2 * gamma * kBT * dt) * db)
            @. x += v * dt
        
            # Correct the position drift (Simulating on the orthogonal is not enough):
            dchi = ISOKANN.dchidx(iso, x)
            phi_val = iso.model(iso.data.featurizer(x))[1]
            error = phi_val - chi_level  
            correction = error / dot(dchi, dchi)
            @. x -= correction * dchi
        
            #Correction for sampling without Fixman Potential 
            # = det(G_M) because dchi is a vector and M⁻1 is a Diagonal matrix with 1/m_i 
            Z = sum(1 ./ m .* dchi.^2)
            Zs[j] = Z
            #Force along chi (Just the Hamiltonian component converges to F too)
            lambdas[j] = -F_proj
        end
        println(iso.model(iso.data.featurizer(x))[1])
  
        # Compute mean force
        mean_forces[i] = mean(lambdas)
        mean_Z[i] = mean(1 ./sqrt.(Zs))
    end
    #sort by chi
    inds= sortperm(chi_vals)
    mean_forces=mean_forces[inds]
    chi_vals=chi_vals[inds]
    mean_Z = mean_Z[inds]
  
    #F_rgd = reverse(integrate_chi(reverse(mean_forces),reverse(chi_vals)))
    F_rgd = integrate_chi(mean_forces,chi_vals)
    F_std =  F_rgd .- kBT*log.(mean_Z)
    p =plot(chi_vals, F_std, xlabel="χ", ylabel="PMF [kj/mol]", legend=false, size=(600,600), frame=:box)
    display(p)
    return F_std
  end
  
  """
  local_mean_force(iso, xs; sim, steps)

Compute the free energy using Thermodynamic Integration. 
Bins the samples into levelsets, computes the mean force along χ locally in
every levelset. (Extremely extensive sampling necessary.)

# Arguments
`iso` the Iso object.
`xs`  the starting points (which should be well distributed in state space).
`nbins` The number of bins/levelsets.
# Returns
`F` the free energy surface of χ in kJ/mol up to an additive constant.
"""
  function local_mean_force(iso::Iso,xs,nbins)
    sim =iso.data.sim
    chi_vals= [ iso.model(iso.data.featurizer(x))[1] for x in eachcol(xs)]
    num_states= size(xs,2)
    inds= sortperm(chi_vals)
    chi_vals = chi_vals[inds]
    # Reorder the columns of xs according to sorted indices
    xs_sorted = xs[:, inds]
  
    bin_size = div(num_states, nbins)
    
    # Preallocate an array to hold the bins (each bin is a submatrix of xs)
    bins = Vector{Matrix{eltype(xs)}}(undef, nbins)
    
    mean_chi_vals=zeros(nbins)
    r = mod(num_states, nbins)
    start_idx = 1
    for i in 1:nbins
      extra = i <= r ? 1 : 0
      end_idx = start_idx + bin_size + extra - 1
      bins[i] = xs_sorted[:, start_idx:end_idx]
      mean_chi_vals[i] = mean(chi_vals[start_idx:end_idx])
      start_idx = end_idx + 1
    end
  
  
    mean_forces = zeros(nbins)
    #mean_Z = zeros(n_bins)
    for i in 1:nbins
      current_bin_size = size(bins[i],2)
      lambdas = zeros(current_bin_size)
      for j in 1:bin_size
        x = bins[i][:,j]
        F = OpenMM.force(sim, x)
        dchi = ISOKANN.dchidx(iso, x)
        F_proj = dot(F, dchi) / dot(dchi, dchi)
        #Correction for sampling without Fixman Potential (do i need this here?)
        # = det(G_M) because dchi is a vector and M⁻1 is a Diagonal matrix with 1/m_i 
        #Z = sum(1 ./ m .* dchi.^2)
        #Zs[j] = Z
  
        #Force along chi (Just the Hamiltonian component converges to F too)
        lambdas[j] = -F_proj
      end
      mean_forces[i]=mean(lambdas)
      #mean_Z[i]=mean(Zs)
    end
  
  
    F_rgd = integrate_chi(mean_forces,mean_chi_vals)
    #F_std =  F_rgd .- kBT*log.(mean_Z)
    p =plot(mean_chi_vals, F_rgd, xlabel="χ", ylabel="PMF [kj/mol]", legend=false, size=(600,600), frame=:box)
    display(p)
    return F_rgd
  end

  """
  integrate_chi(f, chi_vals)

Cumulative integral of the mean force with respect to χ using the trapezoid rule.

# Arguments
`f` The mean force.
`chi_vals` The levelset χ values.
# Returns
`F` the (rigid) free energy surface of χ.
"""
 function integrate_chi(f, chi_vals)
    n = length(chi_vals)
    F = zeros(n)
    # Use the trapezoidal rule; F[1] is set to zero as reference.
    for i in 2:n
        dχ = chi_vals[i] - chi_vals[i-1]
        F[i] = F[i-1] + 0.5 * (f[i] + f[i-1]) * dχ
    end
    return F
end

    """
    delta_G(PMF,chi_vals)
Convenience function to compute free energy differences in a double well free energy surface.
"""
function delta_G(PMF,chi_vals)
    chi_vals = sort(chi_vals)
   
    G0 = minimum(PMF[chi_vals.<0.5])
    G1 = minimum(PMF[chi_vals.>=0.5])
    return G0-G1
end

    """
    function sample_coords(iso,n_points;xs)
Convenience function to uniformly sample npoints out of the χ distribution of xs coordinates.
"""
function sample_coords(iso::Iso,n_points;xs=hcat(iso.data.coords[1],iso.data.coords[2][:,:,1]))
    chi_vals = cpu([iso.model(iso.data.featurizer(x))[1] for x in eachcol(xs)])
    # Determine the range of χ values.
    chi_min, chi_max = minimum(chi_vals), maximum(chi_vals)
    # Create n_points uniformly spaced target χ values.
    target_chis = range(chi_min, stop=chi_max, length=n_points)
    
    # For each target χ, find the index in chi_vals with the minimum absolute difference.
    indices = map(t -> argmin(abs.(chi_vals .- t)), target_chis)
    
    # Extract the corresponding columns from xs.
    sampled_coords = xs[:, indices]
    return sampled_coords
end
  
  
