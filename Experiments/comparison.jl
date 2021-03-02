# load libraries
using ForneyLab
using FL_ComplexNormal
using FL_GaussianScale
using ProgressMeter
using JLD
using PyPlot

# auxiliary functions
pad(sym::Symbol, t::Int) = sym*:_*Symbol(lpad(t,3,'0'))

# function that creates a Forney-style factor graph
function create_model(T::Type, nr_samples::Int64)

    # creat factor graph
    fg = FactorGraph()

    # prior on states
    @RV μ_ξ_0
    @RV Λ_ξ_0
    @RV [id=:ξ_000] ξ_0 ~ GaussianMeanPrecision(μ_ξ_0, Λ_ξ_0)
    placeholder(μ_ξ_0, :μ_ξ_000) 
    placeholder(Λ_ξ_0, :Λ_ξ_000)
    
    # memory vectors
    ξ = Array{Variable}(undef, nr_samples)
    X = Array{Variable}(undef, nr_samples)
    Y = Array{Variable}(undef, nr_samples)
    
    # state transition and observation model
    ξ_t_min = ξ_0
    for t = 1:nr_samples
    
        # state transition
        @RV [id=pad(:ξ,t)] ξ[t] ~ GaussianMeanPrecision(ξ_t_min, placeholder(pad(:Λ_ξ, t)))
    
        # domain transformation
        @RV [id=pad(:X,t)] X[t] ~ GaussianScale{T}(ξ[t])
    
        # observation model
        @RV Y[t] ~ ComplexNormal(X[t], placeholder(pad(:Γ_Y, t)), 0.0.+0.0im)
        placeholder(Y[t], :Y, index=t)
    
        # reset state
        ξ_t_min = ξ[t]
    
    end

    # return graph
    return fg

end

# This function generates the algorithm for a factor graph
function generate_algorithm(fg::FactorGraph, nr_samples::Int64)

    # get variables from factor graphs
    vars = fg.variables

    # define approximate posterior factorization
    q = PosteriorFactorization()
    q_X = Vector{PosteriorFactor}(undef, nr_samples)
    q_ξ = Vector{PosteriorFactor}(undef, nr_samples)
    for t = 1:nr_samples
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

end

# This function performs inference on the model
function perform_inference(data_Y::Array{Complex{Float64},1}, nr_samples::Int64)

    # specify priors
    prior_μ_ξ_0 = 0.0
    prior_Λ_ξ_0 = 1e0
    prior_Λ_ξ = 1e0
    prior_Γ_Y = 1e-6+0.0im

    # initialize marginals dictionary
    marginals = Dict()
        
    # prepare data and prior statistics
    data = Dict(:Y          => data_Y,
                :μ_ξ_000    => prior_μ_ξ_0,
                :Λ_ξ_000    => prior_Λ_ξ_0)
    for t = 1:nr_samples
        data[pad(:Λ_ξ,t)] = prior_Λ_ξ
        data[pad(:Γ_Y,t)] = prior_Γ_Y
    end
    
    # initialize marginals dictionary
    marginals = Dict{Symbol, ProbabilityDistribution}(:ξ_000 => ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=prior_μ_ξ_0, w=prior_Λ_ξ_0))
    for t = 1:nr_samples
        marginals[pad(:ξ, t)] = ProbabilityDistribution(Univariate, GaussianMeanPrecision, m=prior_μ_ξ_0, w=prior_Λ_ξ_0)
    end

    # iterate the variational message passing algorithms
    nr_its = 10
    F = Array{Float64,1}(undef, nr_its)
    @showprogress for it = 1:nr_its

        # perform updates
        for t = 1:nr_samples
            marginals = Base.invokelatest(step!, :X_*t, data, marginals)
        end
        for t = 1:nr_samples
            marginals = Base.invokelatest(step!, :ξ_*t, data, marginals)
        end
        
        # calculate variational Free energy
        F[it] = Base.invokelatest(freeEnergy, data, marginals)
    end 

    # Extract posterior statistics
    m_ξ = Array{Float64,1}(undef, nr_samples)
    v_ξ = Array{Float64,1}(undef, nr_samples)
    for t = 1:nr_samples
        m_ξ[t] = mean(marginals[pad(:ξ,t)])
        v_ξ[t] = cov(marginals[pad(:ξ,t)])
    end

    # return posterior latent states and free energy
    return m_ξ, v_ξ, F

end


# load data
data = load("Experiments/data/synthetic_data.jld")
data_Y = data["freq"]
data_ξ = data["logpower"]
nr_samples = length(data_Y)

# do inference for the VB quadrature model
fg_vb_quad = create_model(VB_Quadrature, nr_samples)
generate_algorithm(fg_vb_quad, nr_samples)
m_ξ_vb_quad, v_ξ_vb_quad, F_vb_quad = perform_inference(data_Y, nr_samples)

# do inference for the VB laplace model
fg_vb_laplace = create_model(VB_LaPlace, nr_samples)
generate_algorithm(fg_vb_laplace, nr_samples)
m_ξ_vb_laplace, v_ξ_vb_laplace, F_vb_laplace = perform_inference(data_Y, nr_samples)

# do inference for the VB vmp model
fg_vb_vmp = create_model(VB_VMP, nr_samples)
generate_algorithm(fg_vb_vmp, nr_samples)
m_ξ_vb_vmp, v_ξ_vb_vmp, F_vb_vmp = perform_inference(data_Y, nr_samples)

# do inference for the BP quadrature model
fg_bp_quad = create_model(BP_Quadrature, nr_samples)
generate_algorithm(fg_bp_quad, nr_samples)
m_ξ_bp_quad, v_ξ_bp_quad, F_bp_quad = perform_inference(data_Y, nr_samples)

# do inference for the BP laplace model
fg_bp_laplace = create_model(BP_LaPlace, nr_samples)
generate_algorithm(fg_bp_laplace, nr_samples)
m_ξ_bp_laplace, v_ξ_bp_laplace, F_bp_laplace = perform_inference(data_Y, nr_samples)

# do inference for the BP VMP model
fg_bp_vmp = create_model(BP_VMP, nr_samples)
generate_algorithm(fg_bp_vmp, nr_samples)
m_ξ_bp_vmp, v_ξ_bp_vmp, F_bp_vmp = perform_inference(data_Y, nr_samples)


# present results (state inference)
_, ax = plt.subplots(nrows=6, figsize=(10,20))

ax[1].plot(data_ξ, color="blue", label="true")
ax[1].plot(m_ξ_bp_quad, color="orange", linewidth=0.5, label="inferred (Sum product, quadrature)")
ax[1].fill_between(0:length(m_ξ_bp_quad)-1, m_ξ_bp_quad - 1*sqrt.(v_ξ_bp_quad), m_ξ_bp_quad + 1*sqrt.(v_ξ_bp_quad), color="orange", alpha=0.5, zorder=5, label="±σ")
ax[1].plot(log.(abs2.(data_Y)), color="black", linestyle="--", label="deterministic")

ax[2].plot(data_ξ, color="blue", label="true")
ax[2].plot(m_ξ_bp_laplace, color="orange", linewidth=0.5, label="inferred (Sum product, Laplace marginal)")
ax[2].fill_between(0:length(m_ξ_bp_laplace)-1, m_ξ_bp_laplace - 1*sqrt.(v_ξ_bp_laplace), m_ξ_bp_laplace + 1*sqrt.(v_ξ_bp_laplace), color="orange", alpha=0.5, zorder=5, label="±σ")
ax[2].plot(log.(abs2.(data_Y)), color="black", linestyle="--", label="deterministic")

ax[3].plot(data_ξ, color="blue", label="true")
ax[3].plot(m_ξ_bp_vmp, color="orange", linewidth=0.5, label="inferred (Sum product, Laplace message)")
ax[3].fill_between(0:length(m_ξ_bp_vmp)-1, m_ξ_bp_vmp - 1*sqrt.(v_ξ_bp_vmp), m_ξ_bp_vmp + 1*sqrt.(v_ξ_bp_vmp), color="orange", alpha=0.5, zorder=5, label="±σ")
ax[3].plot(log.(abs2.(data_Y)), color="black", linestyle="--", label="deterministic")

ax[4].plot(data_ξ, color="blue", label="true")
ax[4].plot(m_ξ_vb_quad, color="orange", linewidth=0.5, label="inferred (Variational, quadrature)")
ax[4].fill_between(0:length(m_ξ_vb_quad)-1, m_ξ_vb_quad - 1*sqrt.(v_ξ_vb_quad), m_ξ_vb_quad + 1*sqrt.(v_ξ_vb_quad), color="orange", alpha=0.5, zorder=5, label="±σ")
ax[4].plot(log.(abs2.(data_Y)), color="black", linestyle="--", label="deterministic")

ax[5].plot(data_ξ, color="blue", label="true")
ax[5].plot(m_ξ_vb_laplace, color="orange", linewidth=0.5, label="inferred (Variational, Laplace marginal)")
ax[5].fill_between(0:length(m_ξ_vb_laplace)-1, m_ξ_vb_laplace - 1*sqrt.(v_ξ_vb_laplace), m_ξ_vb_laplace + 1*sqrt.(v_ξ_vb_laplace), color="orange", alpha=0.5, zorder=5, label="±σ")
ax[5].plot(log.(abs2.(data_Y)), color="black", linestyle="--", label="deterministic")

ax[6].plot(data_ξ, color="blue", label="true")
ax[6].plot(m_ξ_vb_vmp, color="orange", linewidth=0.5, label="inferred (Variational, Laplace message)")
ax[6].fill_between(0:length(m_ξ_vb_vmp)-1, m_ξ_vb_vmp - 1*sqrt.(v_ξ_vb_vmp), m_ξ_vb_vmp + 1*sqrt.(v_ξ_vb_vmp), color="orange", alpha=0.5, zorder=5, label="±σ")
ax[6].plot(log.(abs2.(data_Y)), color="black", linestyle="--", label="deterministic")

ax[1].set_xlabel("sample"), ax[2].set_xlabel("sample"), ax[3].set_xlabel("sample"), ax[4].set_xlabel("sample"), ax[5].set_xlabel("sample"), ax[6].set_xlabel("sample")
ax[1].set_ylabel("latent state ξ"), ax[2].set_ylabel("latent state ξ"), ax[3].set_ylabel("latent state ξ"), ax[4].set_ylabel("latent state ξ"), ax[5].set_ylabel("latent state ξ"), ax[6].set_ylabel("latent state ξ")
ax[1].grid(), ax[2].grid(), ax[3].grid(), ax[4].grid(), ax[5].grid(), ax[6].grid()
ax[1].legend(loc="lower left"), ax[2].legend(loc="lower left"), ax[3].legend(loc="lower left"), ax[4].legend(loc="lower left"), ax[5].legend(loc="lower left"), ax[6].legend(loc="lower left")
ax[1].set_xlim(0, nr_samples), ax[2].set_xlim(0, nr_samples), ax[3].set_xlim(0, nr_samples), ax[4].set_xlim(0, nr_samples), ax[5].set_xlim(0, nr_samples), ax[6].set_xlim(0, nr_samples)
plt.tight_layout()
plt.gcf()
plt.savefig("./Experiments/exports/state_inference.png")
plt.savefig("./Experiments/exports/state_inference.eps")
plt.savefig("./Experiments/exports/state_inference.pdf")

# present results (free energy comparison)
plt.figure()
plt.plot(1:length(F_bp_quad), F_bp_quad/nr_samples, label="Sum product, quadrature")
plt.plot(1:length(F_bp_laplace), F_bp_laplace/nr_samples, label="Sum product, Laplace marginal")
plt.plot(1:length(F_bp_vmp), F_bp_vmp/nr_samples, label="Sum product, Laplace message")
plt.plot(1:length(F_vb_quad), F_vb_quad/nr_samples, label="Variational, quadrature")
plt.plot(1:length(F_vb_laplace), F_vb_laplace/nr_samples, label="Variational, Laplace marginal")
plt.plot(1:length(F_vb_vmp), F_vb_vmp/nr_samples, label="Variational, Laplace message")
plt.grid()
plt.xlim(1, length(F_vb_quad))
plt.ylim(3, 4.5)
plt.legend(loc="upper left")
plt.ylabel("normalized variational free energy [nats/sample]")
plt.xlabel("iteration")

zoomax1 = plt.axes([.6, .375, .275, .1])
zoomax1.plot(1:length(F_bp_quad), F_bp_quad/nr_samples)
zoomax1.plot(1:length(F_bp_laplace), F_bp_laplace/nr_samples)
zoomax1.plot(1:length(F_bp_vmp), F_bp_vmp/nr_samples)
zoomax1.plot(1:length(F_vb_quad), F_vb_quad/nr_samples)
zoomax1.plot(1:length(F_vb_laplace), F_vb_laplace/nr_samples)
zoomax1.plot(1:length(F_vb_vmp), F_vb_vmp/nr_samples)
plt.grid()
plt.xlim(9.9, length(F_vb_quad))
plt.ylim(3.060869, 3.06088)
plt.setp(zoomax1, yticks=[])
plt.xticks([9.9, 9.95, 10])

zoomax2 = plt.axes([.6, .55, .275, .1])
zoomax2.plot(1:length(F_bp_quad), F_bp_quad/nr_samples)
zoomax2.plot(1:length(F_bp_laplace), F_bp_laplace/nr_samples)
zoomax2.plot(1:length(F_bp_vmp), F_bp_vmp/nr_samples)
zoomax2.plot(1:length(F_vb_quad), F_vb_quad/nr_samples)
zoomax2.plot(1:length(F_vb_laplace), F_vb_laplace/nr_samples)
zoomax2.plot(1:length(F_vb_vmp), F_vb_vmp/nr_samples)
plt.grid()
plt.xlim(9.9, length(F_vb_quad))
plt.ylim(3.08355, 3.08361)
plt.setp(zoomax2, yticks=[])
plt.xticks([9.9, 9.95, 10])

zoomax3 = plt.axes([.6, .725, .275, .1])
zoomax3.plot(1:length(F_bp_quad), F_bp_quad/nr_samples)
zoomax3.plot(1:length(F_bp_laplace), F_bp_laplace/nr_samples)
zoomax3.plot(1:length(F_bp_vmp), F_bp_vmp/nr_samples)
zoomax3.plot(1:length(F_vb_quad), F_vb_quad/nr_samples)
zoomax3.plot(1:length(F_vb_laplace), F_vb_laplace/nr_samples)
zoomax3.plot(1:length(F_vb_vmp), F_vb_vmp/nr_samples)
plt.grid()
plt.xlim(9.9, length(F_vb_quad))
plt.ylim(3.3091, 3.3093)
plt.setp(zoomax3, yticks=[])
plt.xticks([9.9, 9.95, 10])

plt.gcf()
plt.savefig("./Experiments/exports/free_energy.png")
plt.savefig("./Experiments/exports/free_energy.eps")
plt.savefig("./Experiments/exports/free_energy.pdf")

# save results
save("Experiments/exports/data/comparison_true.jld", "data", data_ξ)
save("Experiments/exports/data/comparison_deterministic.jld", "data", log.(abs2.(data_Y)))
save("Experiments/exports/data/comparison_sp_quad.jld", "mean", m_ξ_bp_quad, "std", sqrt.(v_ξ_bp_quad), "FE", F_bp_quad)
save("Experiments/exports/data/comparison_sp_laplace.jld", "mean", m_ξ_bp_laplace, "std", sqrt.(v_ξ_bp_laplace), "FE", F_bp_laplace)
save("Experiments/exports/data/comparison_sp_vmp.jld", "mean", m_ξ_bp_vmp, "std", sqrt.(v_ξ_bp_vmp), "FE", F_bp_vmp)
save("Experiments/exports/data/comparison_vb_quad.jld", "mean", m_ξ_vb_quad, "std", sqrt.(v_ξ_vb_quad), "FE", F_vb_quad)
save("Experiments/exports/data/comparison_vb_laplace.jld", "mean", m_ξ_vb_laplace, "std", sqrt.(v_ξ_vb_laplace), "FE", F_vb_laplace)
save("Experiments/exports/data/comparison_vb_vmp.jld", "mean", m_ξ_vb_vmp, "std", sqrt.(v_ξ_vb_vmp), "FE", F_vb_vmp)

