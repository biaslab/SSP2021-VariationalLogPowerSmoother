# load packages and files
using PyPlot
using JLD
include("src/data.jl")
include("src/graph.jl")
include("src/inference.jl")

# load data
yi, Yi, logY2i = load_data("Experiments/data/woman.wav", 5; noise_db=-80, n=32, noverlap=0, fs=16000)

# build graph
fg = create_model(size(Yi,2), size(Yi,1))

# generate algorithm
code = generate_algorithm(fg, size(Yi,2))

# perform inference
ξ_inferred, _ = perform_inference(yi, Yi, -80; fs=16000, nr_its=10)

# plot results
_, ax = plt.subplots(nrows=2, figsize=(10,10))
img = ax[1].imshow(logY2i[2:end-1,:], aspect="auto", origin="lower", cmap="jet")
plt.colorbar(img, ax=ax[1])
img = ax[2].imshow(ξ_inferred, aspect="auto", origin="lower", cmap="jet")
plt.colorbar(img, ax=ax[2])
ax[1].set_xlabel("frame"), ax[2].set_xlabel("frame")
ax[1].set_ylabel("frequency bin"), ax[2].set_ylabel("frequency bin")
ax[1].set_title("deterministic log-power spectrum")
ax[2].set_title("Inferred values of ξ")
plt.gcf()
plt.savefig("Experiments/exports/tracking.png")
plt.savefig("Experiments/exports/tracking.eps")
plt.savefig("Experiments/exports/tracking.pdf")

# save data
save("Experiments/exports/data/tracking_true.jld", "data", logY2i[2:end-1,:])
save("Experiments/exports/data/tracking_inferred.jld", "data", ξ_inferred)
