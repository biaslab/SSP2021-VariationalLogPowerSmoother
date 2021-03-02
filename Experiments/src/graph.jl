using ForneyLab
using FL_ComplexNormal
using FL_ComplexToReal
using FL_GaussianScale

# auxiliary functions
pad(sym::Symbol, t::Int) = sym*:_*Symbol(lpad(t,3,'0'))

# function that creates a Forney-style factor graph
function create_model(nr_frames::Int64, nr_freqs::Int64)

    nr_samples = 2*(nr_freqs-1)

    # creat factor graph
    fg = FactorGraph()

    # prior on states
    @RV μ_ξ_0
    @RV Λ_ξ_0
    @RV [id=:ξ_000] ξ_0 ~ GaussianMeanPrecision(μ_ξ_0, Λ_ξ_0)
    placeholder(μ_ξ_0, :μ_ξ_000, dims=(nr_freqs,)) 
    placeholder(Λ_ξ_0, :Λ_ξ_000, dims=(nr_freqs, nr_freqs))
    
    # memory vectors
    ξ = Array{Variable}(undef, nr_frames)
    X = Array{Variable}(undef, nr_frames)
    R = Array{Variable}(undef, nr_frames)
    C = Array{Variable}(undef, nr_frames)
    x = Array{Variable}(undef, nr_frames)
    y = Array{Variable}(undef, nr_frames)
    
    # state transition and observation model
    ξ_t_min = ξ_0
    for t = 1:nr_frames
    
        # state transition
        @RV [id=pad(:ξ,t)] ξ[t] ~ GaussianMeanPrecision(ξ_t_min, placeholder(pad(:Λ_ξ, t), dims=(nr_freqs, nr_freqs)))
    
        # domain transformation
        @RV [id=pad(:X,t)] X[t] ~ GaussianScaleMV(ξ[t])

        # reparameterize coefficients
        @RV [id=pad(:R,t)] R[t] ~ ComplexToReal(X[t])

        # fourier decomposition
        @RV [id=pad(:C,t)] C[t]
        placeholder(C[t], pad(:C,t), dims=(nr_samples, nr_freqs*2))
        @RV [id=pad(:x,t)] x[t] = C[t] * R[t]
    
        # observation model
        @RV y[t] ~ GaussianMeanPrecision(x[t], placeholder(pad(:Λ_y, t), dims=(nr_samples, nr_samples)))
        placeholder(y[t], pad(:y,t), dims=(nr_samples,))
    
        # reset state
        ξ_t_min = ξ[t]
    
    end

    # return graph
    return fg

end

# This function generates the algorithm for a factor graph
function generate_algorithm(fg::FactorGraph, nr_frames::Int64)

    # get variables from factor graphs
    vars = fg.variables

    # define approximate posterior factorization
    q = PosteriorFactorization()
    q_X = Vector{PosteriorFactor}(undef, nr_frames)
    q_ξ = Vector{PosteriorFactor}(undef, nr_frames)
    for t = 1:nr_frames
        q_ξ[t] = PosteriorFactor(vars[pad(:ξ,t)],id=:ξ_*t)
        q_X[t] = PosteriorFactor(vars[pad(:X,t)],id=:X_*t)
    end
    q_ξ_0 = PosteriorFactor(vars[:ξ_000], id=:ξ_000)

    # generate algorithm
    algo = messagePassingAlgorithm(free_energy=true)

    # convert algorithm to code
    code = algorithmSourceCode(algo, free_energy=true)

    # parse code 
    eval(Meta.parse(code))

    # return code
    return code

end


function calc_C(fs::Int64, nr_freqs::Int64, t::Array{Float64})
    # Info:     This function calculates the Fourier series matrix C, which
    #           decomposes a time-domain signal into its real Fourier coefficients.
    #
    # Inputs:   f - array with center frequencies of Fourier coefficients
    #           t - time stamps for sinusoidal functions in the C-matrix
    #
    # Outputs:  C - Matrix of dimensions (N, 2M), where N represents the amount of
    #               time instances and where M represents the amount of frequency
    #               bins, which creates the Fourier decomposition
    
    f = collect(1:nr_freqs)/(nr_freqs+1)/2*fs

    # allocate space for matrix C
    C = Array{Float64}(undef, length(t), 2*length(f))

    # loop through time (rows) and fill matrix C
    for (idx, ti) in enumerate(t)
        C[idx, :] = cat(dims=1,sin.(2*pi*f*ti), cos.(2*pi*f*ti))
    end

    # return matrix C
    return C
end;