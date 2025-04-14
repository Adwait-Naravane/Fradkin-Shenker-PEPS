using Pkg
Pkg.instantiate()

# # Get command-line arguments: hx, hz, χ, D
# if length(ARGS) < 4
#     error("Usage: julia tests.jl <hx> <hz> <χ> <D>")
# end

# hx = parse(Float64, ARGS[1])
# hz = parse(Float64, ARGS[2])
# χ = parse(Int, ARGS[3])
# D = parse(Int, ARGS[4])

# println("Running with: hx=$hx, hz=$hz, χ=$χ, D=$D")
# include("new_toolbox.jl")



file = jldopen("final_Psi_trivial_hx=$(hx)_hz=$(hz)_χ=$(χ)_D=$(D).jld2", "r")
new_Ψ = file["Ψ"]
env = file["env"]
E = file["E"]
convhistory = file["convhistory"]
close(file)

Z = partition_function_peps(new_Ψ)
env_Z = get_new_environment_Z(env, new_Ψ)


ξ = correlation_length(Z, env_Z)