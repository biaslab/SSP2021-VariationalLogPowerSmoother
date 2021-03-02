using ProgressMeter
using LinearAlgebra

function perform_inference(yi::Array{Float64,1}, Yi::Array{Complex{Float64},2}, noise_db::Int64; fs::Int64=16000, nr_its::Int64=10)

    # fetch information from data
    nr_frames = size(Yi, 2)
    nr_freqs = size(Yi, 1) - 2
    nr_samples = (nr_freqs + 1) * 2
    y_noisy = yi
        
    # prepare data dictionary
    data = Dict()
    data[:μ_ξ_000] = -5*ones(nr_freqs)
    data[:Λ_ξ_000] = diagm(ones(nr_freqs))
    for k = 1:nr_frames
        data[pad(:Λ_ξ, k)] = diagm(1e-3*ones(nr_freqs))
        data[pad(:C,k)] = calc_C(fs, nr_freqs, collect((k-1)*nr_samples+1:k*nr_samples)/fs)
        data[pad(:Λ_y, k)] = diagm(1/(10^(noise_db/10))*ones(nr_samples))
        data[pad(:y,k)] = y_noisy[(k-1)*nr_samples+1:k*nr_samples] .- mean(y_noisy[(k-1)*nr_samples+1:k*nr_samples])
    end

    # prepare marginals dictionary
    marginals = Dict()
    marginals[:ξ_000] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=data[:μ_ξ_000], w=diagm(ones(nr_freqs)))
    for k = 1:nr_frames
        marginals[pad(:ξ,k)] = ProbabilityDistribution(Multivariate, GaussianMeanPrecision, m=zeros(nr_freqs), w=diagm(ones(nr_freqs)))
    end

    # loop through iterations
    @showprogress for it = 1:nr_its

            # perform updates
            for t = 1:nr_frames
                marginals = Base.invokelatest(step!, :X_*t, data, marginals)
            end
            for t = 1:nr_frames
                marginals = Base.invokelatest(step!, :ξ_*t, data, marginals)
            end

    end

    # fetch results
    ξ_mem = Array{Float64,2}(undef, nr_freqs, nr_frames)
    x_mem = Array{Float64,1}(undef, nr_frames*nr_samples)
    for k = 1:nr_frames 
        ξ_mem[:,k] = ForneyLab.unsafeMean(marginals[pad(:ξ,k)])
        x_mem[(k-1)*nr_samples+1:k*nr_samples] = ForneyLab.unsafeMean(marginals[pad(:x,k)]) .+ mean(y_noisy[(k-1)*nr_samples+1:k*nr_samples])
    end

    # return results
    return ξ_mem, x_mem

end