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
include("new_toolbox.jl")

using JLD2, FileIO, Glob, CSV, DataFrames

# Path to folder
folder = "Saved_content"
files = glob("final_Psi_trivial_1e4*_hx=*_hz=*_χ=*_D=*.jld2", folder)

results = DataFrame(
    hx = Float64[],
    hz = Float64[],
    chi = Int[],
    D = Int[],
    E = [],
    ξv = Float64[],
    ξh = Float64[],
    infinite_tHooft = ComplexF64[],
    infinite_Wilson = ComplexF64[]
)

for file in files
    filename = split(basename(file), ".jld2")[1]
    println("Processing $filename")

    try
        # Extract params using regex
        m = match(r"hx=([0-9.eE+-]+)_hz=([0-9.eE+-]+)_χ=([0-9]+)_D=([0-9]+)", filename)
        if m === nothing || any(x -> x === nothing, m.captures)
            @warn "Could not extract parameters from filename: $filename"
            continue
        end
        hx = parse(Float64, m.captures[1])
        hz = parse(Float64, m.captures[2])
        chi = parse(Int, m.captures[3])
        D = parse(Int, m.captures[4])

        # Load file
        f = jldopen(file, "r")
        Ψ = f["Ψ"]
        E = f["E"]
        convhistory = f["convhistory"]
        env = f["env"]
        close(f)

        # Compute quantities
        Z = partition_function_peps(Ψ)
        env_Z = get_new_environment_Z(env, Ψ)
        ξv, ξh,  = correlation_length(Z, env_Z)

        vals_tHooft_trivial, vals_tHooft, vals_Wilson_trivial, vals_Wilson = strings_CTMRG(Ψ, env)
        infinite_tHooft = vals_tHooft[1] / vals_tHooft_trivial[1]
        infinite_Wilson = vals_Wilson[1] / vals_Wilson_trivial[1]

        # Append to results
        push!(results, (hx, hz, chi, D, E, ξv..., ξh..., infinite_tHooft, infinite_Wilson))

    catch e
        @warn "Skipping $file due to error" exception = e
    end
end

# Save to CSV
CSV.write("summary_results.csv", results)
# D = 4
# χ = 36
# hx = 0.3
# hz = 0.2


# file = jldopen("Saved_content/final_Psi_trivial_1e4_hx=$(hx)_hz=$(hz)_χ=$(χ)_D=$(D).jld2", "r")
# new_Ψ = file["Ψ"]
# env = file["env"]
# E = file["E"]
# convhistory = file["convhistory"]
# close(file)

# Z = partition_function_peps(new_Ψ)
# env_Z = get_new_environment_Z(env, new_Ψ)
# ξv, ξh, = correlation_length(Z, env_Z)

# vals_tHooft_trivial, vals_tHooft, vals_Wilson_trivial, vals_Wilson = strings_CTMRG(new_Ψ, env)

# infinite_tHfoot = vals_tHooft[1]/vals_tHooft_trivial[1]
# infinite_Wilson = vals_Wilson[1]/vals_Wilson_trivial[1]