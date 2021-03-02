"""
This file generates data for noisy complex frequency coefficients over time, modelled by a Gaussian scale model, modelled by a Gaussian random walk.
"""

using JLD
using PyPlot

# settings
nr_samples = 100

# generate log-power coefficients
ξ = Array{Float64,1}(undef, nr_samples)
ξ[1] = randn()
for n = 2:nr_samples
    ξ[n] = ξ[n-1] + sqrt(1e0)*randn()
end

# generate complex coefficients
X = Array{Complex{Float64},1}(undef, nr_samples)
for n = 1:nr_samples
    X[n] = sqrt(0.5*exp(ξ[n]))*randn() + sqrt(0.5*exp(ξ[n]))*randn()*1im
end

# generate output
Y = Array{Complex{Float64},1}(undef, nr_samples)
for n = 1:nr_samples
    Y[n] = X[n] + sqrt(0.5*1e-6)*randn() + sqrt(0.5*1e-6)*randn()*1im
end

# save data
save("Experiments/data/synthetic_data.jld", "freq", Y, "logpower", ξ)


# plot states
plt.figure()
plt.plot(ξ)
plt.grid()
plt.gcf()