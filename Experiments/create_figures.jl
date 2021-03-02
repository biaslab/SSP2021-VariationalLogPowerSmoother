using JLD, PyPlot #Plots, PGFPlotsX, LaTeXStrings
# using LaTeXStrings

rcParams = PyPlot.PyDict(PyPlot.matplotlib."rcParams")
rcParams["text.usetex"] = true
rcParams["font.size"] = 8
rcParams["font.family"] = "lmodern"
rcParams["axes.labelpad"]=1


# load data
speech_logX2 = load("Experiments/exports/data/tracking_true.jld")["data"]
speech_ξ = load("Experiments/exports/data/tracking_inferred.jld")["data"]
sp_quad = load("Experiments/exports/data/comparison_sp_quad.jld")
sp_laplace = load("Experiments/exports/data/comparison_sp_laplace.jld")
sp_vmp = load("Experiments/exports/data/comparison_sp_vmp.jld")
vb_quad = load("Experiments/exports/data/comparison_vb_quad.jld")
vb_laplace = load("Experiments/exports/data/comparison_vb_laplace.jld")
vb_vmp = load("Experiments/exports/data/comparison_vb_vmp.jld")


# middle column (FE comparison)
fig = plt.figure()
fig.set_size_inches(7/3*2,9/2/3*2) 
plt.plot(1:length(sp_quad["FE"]), sp_quad["FE"], label="Sum-product, quadrature")
plt.plot(1:length(sp_laplace["FE"]), sp_laplace["FE"], label="Sum-product, Laplace marginal")
plt.plot(1:length(sp_vmp["FE"]), sp_vmp["FE"], label="Sum-product, Laplace message")
plt.plot(1:length(vb_quad["FE"]), vb_quad["FE"], label="Variational, quadrature")
plt.plot(1:length(vb_laplace["FE"]), vb_laplace["FE"], label="Variational, Laplace marginal")
plt.plot(1:length(vb_vmp["FE"]), vb_vmp["FE"], label="Variational, Laplace message")
plt.grid()
plt.xlim(1, length(sp_quad["FE"]))
plt.ylim(300, 450)
plt.legend(loc="upper left")
plt.ylabel("Variational free energy [nats]")
plt.xlabel("iteration")
plt.tight_layout()

zoomax1 = plt.axes([.725, .4, .2, .1])
zoomax1.plot(1:length(sp_quad["FE"]), sp_quad["FE"])
zoomax1.plot(1:length(sp_laplace["FE"]), sp_laplace["FE"])
zoomax1.plot(1:length(sp_vmp["FE"]), sp_vmp["FE"])
zoomax1.plot(1:length(vb_quad["FE"]), vb_quad["FE"])
zoomax1.plot(1:length(vb_laplace["FE"]), vb_laplace["FE"])
zoomax1.plot(1:length(vb_vmp["FE"]), vb_vmp["FE"])
plt.grid()
plt.xlim(9.9, length(sp_quad["FE"]))
plt.ylim(3.060869*100, 3.06088*100)
#plt.setp(zoomax1, yticks=[])
plt.xticks([9.9, 9.95, 10], fontsize=7)
plt.yticks(fontsize=7)
zoomax1.yaxis.offsetText.set_fontsize(7)

zoomax2 = plt.axes([.725, .6, .2, .1])
zoomax2.plot(1:length(sp_quad["FE"]), sp_quad["FE"])
zoomax2.plot(1:length(sp_laplace["FE"]), sp_laplace["FE"])
zoomax2.plot(1:length(sp_vmp["FE"]), sp_vmp["FE"])
zoomax2.plot(1:length(vb_quad["FE"]), vb_quad["FE"])
zoomax2.plot(1:length(vb_laplace["FE"]), vb_laplace["FE"])
zoomax2.plot(1:length(vb_vmp["FE"]), vb_vmp["FE"])
plt.grid()
plt.xlim(9.9, length(sp_quad["FE"]))
plt.ylim(3.08355*100, 3.08361*100)
#plt.setp(zoomax2, yticks=[])
plt.xticks([9.9, 9.95, 10], fontsize=7)
plt.yticks(fontsize=7)
zoomax2.yaxis.offsetText.set_fontsize(7)

zoomax3 = plt.axes([.725, .8, .2, .1])
zoomax3.plot(1:length(sp_quad["FE"]), sp_quad["FE"])
zoomax3.plot(1:length(sp_laplace["FE"]), sp_laplace["FE"])
zoomax3.plot(1:length(sp_vmp["FE"]), sp_vmp["FE"])
zoomax3.plot(1:length(vb_quad["FE"]), vb_quad["FE"])
zoomax3.plot(1:length(vb_laplace["FE"]), vb_laplace["FE"])
zoomax3.plot(1:length(vb_vmp["FE"]), vb_vmp["FE"])
plt.grid()
plt.xlim(9.9, length(sp_quad["FE"]))
plt.ylim(3.3091*100, 3.3093*100)
#plt.setp(zoomax3, yticks=[])
plt.xticks([9.9, 9.95, 10], fontsize=7)
plt.yticks(fontsize=7)
zoomax3.yaxis.offsetText.set_fontsize(7)


plt.tight_layout()
plt.gcf()
plt.savefig("./Experiments/exports/free_energy.png", bbox_inches="tight")
plt.savefig("./Experiments/exports/free_energy.eps", bbox_inches="tight")
plt.savefig("./Experiments/exports/free_energy.pdf", bbox_inches="tight")

# Plot of tracked spectrograms
_, ax = plt.subplots(ncols=2, figsize=(7/3*2,9/2/3))
img = ax[1].imshow(speech_logX2, aspect="auto", origin="lower", cmap="jet", extent=[0,5,0,15])
plt.colorbar(img, ax=ax[1])
img = ax[2].imshow(speech_ξ, aspect="auto", origin="lower", cmap="jet", extent=[0,5,0,15])
plt.colorbar(img, ax=ax[2])
ax[1].set_xlabel("time [sec]"), ax[2].set_xlabel("time [sec]")
ax[1].set_ylabel("frequency bin"), ax[2].set_ylabel("frequency bin")
ax[1].set_title("Deterministic spectrogram")
ax[2].set_title(L"Inferred values of $\xi$")
plt.tight_layout(pad=0, w_pad=0.2)
plt.gcf()

plt.savefig("./Experiments/exports/tracking.png", bbox_inches="tight")
plt.savefig("./Experiments/exports/tracking.eps", bbox_inches="tight")
plt.savefig("./Experiments/exports/tracking.pdf", bbox_inches="tight")

# # initial settings
# pgfplotsx()
# push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepgfplotslibrary{fillbetween}")
# push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usetikzlibrary{spy}")

# # plot 1
# n = (1:size(speech_logX2,2)) .* 32 / 16000
# f = 1:size(speech_logX2,1)
# c = Coordinates(n, f, speech_logX2')
# figure = @pgf Axis(
#     {
#         view = (0, 90),
#         colorbar,
#         "colormap/jet",
#         ylabel="frequency bin",
#         xlabel="time [sec]",
#         title="deterministic spectrogram"
#     },
#     Plot3(
#         {
#             surf,
#             shader = "flat",
#         },
#         c)
# )
# pgfsave("Experiments/exports/tikzfigures/speech_deterministic.tikz", figure)

# # plot 2
# n = (1:size(speech_ξ,2)) .* 32 / 16000
# f = 1:size(speech_ξ,1)
# c = Coordinates(n, f, speech_ξ')
# figure = @pgf Axis(
#     {
#         view = (0, 90),
#         colorbar,
#         "colormap/jet",
#         ylabel="frequency bin",
#         xlabel="time [sec]",
#         title="inferred values of "*L"\bm{\xi}"
#     },
#     Plot3(
#         {
#             surf,
#             shader = "flat",
#         },
#         c)
# )
# pgfsave("Experiments/exports/tikzfigures/speech_inferred.tikz", figure)

# # plot 3
# figure = @pgf Axis(
#     {
#         ylabel="variational free energy [nats]",
#         xlabel="iteration",
#         grid="major",
#         xmin = 1,
#         xmax = 10,
#         ymax=500,
#         no_markers
#     },
#     Plot(Coordinates(collect(1:length(sp_quad["FE"])),sp_quad["FE"])),
#     LegendEntry("Sum-product - Quadrature"),
#     Plot(Coordinates(collect(1:length(sp_laplace["FE"])),sp_laplace["FE"])),
#     LegendEntry("Sum-product - Laplace marginal"),
#     Plot(Coordinates(collect(1:length(sp_vmp["FE"])),sp_vmp["FE"])),
#     LegendEntry("Sum-product - Laplace message"),
#     Plot(Coordinates(collect(1:length(vb_quad["FE"])),vb_quad["FE"])),
#     LegendEntry("Variational - Quadrature"),
#     Plot(Coordinates(collect(1:length(vb_laplace["FE"])),vb_laplace["FE"])),
#     LegendEntry("Variational - Laplace marginal"),
#     Plot(Coordinates(collect(1:length(vb_vmp["FE"])),vb_vmp["FE"])),
#     LegendEntry("Variational - Laplace message"),
# )
# lens!([1, 6], [0.9, 1.1], inset = (1, bbox(0.5, 0.0, 0.4, 0.4)))
# pgfsave("Experiments/exports/tikzfigures/comparison_free_energy.tikz", figure)