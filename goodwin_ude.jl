
#using Pkg

# 1. Add SciML registry (required for DataDrivenDiffEq and related tools)
#Pkg.Registry.add(Pkg.Registry.RegistrySpec(url = "https://github.com/SciML/Registry.git"))
#Pkg.Registry.add("SciML")


# 2. Install all required packages in one step
# Pkg.add([
#     "OrdinaryDiffEq",
#     "ModelingToolkit",
#     "DataDrivenDiffEq",
#     "SciMLSensitivity",
#     "DataDrivenSparse",
#     "Optimization",
#     "OptimizationOptimisers",
#     "OptimizationOptimJL",
#     "LineSearches",
#     "ComponentArrays",
#     "Lux",
#     "Zygote",
#     "Plots",
#     "StableRNGs"
# ])

#Pkg.add(["DataFrames", "CSV"]) # I added this for Python exportation of the data

# Now import the packages (after installation)
using OrdinaryDiffEq, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using LinearAlgebra, Statistics
using ComponentArrays, Lux, Zygote, Plots, StableRNGs
using DataFrames, CSV

const ODE = OrdinaryDiffEq
const SMS = SciMLSensitivity
const OPT = Optimization

# Set a random seed for reproducible behaviour
rng = StableRNGs.StableRNG(1111)

function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Define the experimental parameter

tspan = (0.0, 10.0)
#u0 = [0.9416, 0.6552]
u0 = [0.92, 0.65]

#u0 = 5.0f0 * rand(rng, 2)
p_ = [1.399, 2.239, 2.57, 2.43]
#p_ = [0.1399, 0.2239, 0.257, 0.243]

#tspan = (0.0, 5.0)
#u0 = 5.0f0 * rand(rng, 2)
#p_ = [1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.25)

# Add noise in terms of the mean
X = Array(solution)
t = solution.t

x̄ = Statistics.mean(X, dims = 2)
noise_magnitude = 1e-3 #here i changed from 5e-3 to 1e-3 to reduce noise
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

p = plot(
    solution,
    alpha = 0.75,
    color = :black,
    label = ["True Data" nothing],
)
scatter!(p, t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])
display(p)

# Definition of the Universal Differential Equation
rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward
const U = Lux.Chain(Lux.Dense(2, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 2))
# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)
const _st = st

# We then define the UDE as a dynamical system that is u' = known(u) + NN(u) like 

# Define the hybrid model
function ude_dynamics!(du, u, p, t, p_true)
    û = U(u, p, _st)[1] # Network prediction
    du[1] = p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)
# Define the problem
prob_nn = ODE.ODEProblem(nn_dynamics!, Xₙ[:, 1], tspan, p)

# Setting Up the Training Loop

function predict(θ, X = Xₙ[:, 1], T = t)
    _prob = ODE.remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(ODE.solve(_prob, ODE.Vern7(), saveat = T,
        abstol = 1e-6, reltol = 1e-6,
        sensealg = SMS.QuadratureAdjoint(autojacvec = SMS.ReverseDiffVJP(true))))
end

function loss(θ)
    X̂ = predict(θ)
    Statistics.mean(abs2, Xₙ .- X̂)
end


losses = Float64[]

callback = function (state, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = OPT.AutoZygote()
optf = OPT.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = OPT.OptimizationProblem(optf, ComponentArrays.ComponentVector{Float64}(p))

res1 = OPT.solve(
    optprob, OptimizationOptimisers.Adam(0.05), callback = callback, maxiters = 5000)
println("Training loss after $(length(losses)) iterations: $(losses[end])") 
# here i changed Adam() to Adam(0.05) to speed up convergence 


optprob2 = OPT.OptimizationProblem(optf, res1.u)
res2 = OPT.solve(
    optprob2, OptimizationOptimJL.BFGS(linesearch = LineSearches.BackTracking()), callback = callback, maxiters = 2000)
println("Final training loss after $(length(losses)) iterations: $(losses[end])")
# here i changed maxiters from 1000 to 2000 to allow more iterations for convergence
# Rename the best candidate
p_trained = res2.u 

# Plot the losses
pl_losses = Plots.Plots.plot(1:5000, losses[1:5000], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "ADAM", color = :blue)
Plots.Plots.plot!(5001:length(losses), losses[5001:end], yaxis = :log10, xaxis = :log10,
    xlabel = "Iterations", ylabel = "Loss", label = "BFGS", color = :red) 
display(pl_losses)


## Analysis of the trained network
# Plot the data and the approximation
ts = first(solution.t):(Statistics.mean(diff(solution.t)) / 2):last(solution.t)
X̂ = predict(p_trained, Xₙ[:, 1], ts)
# Trained on noisy data vs real solution
pl_trajectory = Plots.Plots.plot(ts, transpose(X̂), xlabel = "t", ylabel = "x(t), y(t)", color = :red,
    label = ["UDE Approximation" nothing])
Plots.scatter!(solution.t, transpose(Xₙ), color = :black, label = ["Measurements" nothing])
display(pl_trajectory) 

# Ideal unknown interactions of the predictor
Ȳ = [-p_[2] * (X̂[1, :] .* X̂[2, :])'; p_[3] * (X̂[1, :] .* X̂[2, :])']
# Neural network guess
Ŷ = U(X̂, p_trained, st)[1]

pl_reconstruction = Plots.plot(ts, transpose(Ŷ), xlabel = "t", ylabel = "U(x,y)", color = :red,
    label = ["UDE Approximation" nothing])
Plots.plot!(ts, transpose(Ȳ), color = :black, label = ["True Interaction" nothing])

# Plot the error
pl_reconstruction_error = Plots.plot(ts, LinearAlgebra.norm.(eachcol(Ȳ - Ŷ)), yaxis = :log, xlabel = "t",
    ylabel = "L2-Error", label = nothing, color = :red)
pl_missing = Plots.plot(pl_reconstruction, pl_reconstruction_error, layout = (2, 1))

pl_overall = Plots.plot(pl_trajectory, pl_missing)
Plots.savefig(pl_overall, "goodwin_overall.png")
 
# =======================
# Export data for Python
# =======================

# Build a DataFrame with everything you might want for SINDy in Python
df = DataFrame(
    t        = ts,
    x_hat    = vec(X̂[1, :]),   # UDE-smoothed state 1
    y_hat    = vec(X̂[2, :]),   # UDE-smoothed state 2
    f1_true  = vec(Ȳ[1, :]),   # true missing term for eqn 1
    f2_true  = vec(Ȳ[2, :]),   # true missing term for eqn 2
    f1_hat   = vec(Ŷ[1, :]),   # NN-approximated missing term for eqn 1
    f2_hat   = vec(Ŷ[2, :])    # NN-approximated missing term for eqn 2
)

# Optionally also store the original noisy data aligned to solution.t
# (this is at coarser time grid, so I keep a separate file)
df_noisy = DataFrame(
    t        = t,               # original time points from `solution`
    x_noisy  = vec(Xₙ[1, :]),
    y_noisy  = vec(Xₙ[2, :])
)

# --------- CSV (recommended; Excel-friendly) ----------
CSV.write("ude_data_sindy_clean.csv", df)
CSV.write("ude_data_noisy.csv", df_noisy)
println("Saved ude_data_sindy_clean.csv and ude_data_noisy.csv")


