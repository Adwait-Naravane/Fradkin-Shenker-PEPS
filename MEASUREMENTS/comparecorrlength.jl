using Pkg
Pkg.activate(".")
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
include("../utility/new_toolbox.jl")

using JLD2, FileIO, Glob, CSV, DataFrames

# Path to folder
folder = "Saved_content"
files = glob("final_Psi_trivial_1e4*_hx=*_hz=*_χ=*_D=*.jld2", folder)

results = DataFrame(
    hx=Float64[],
    hz=Float64[],
    chi=Int[],
    D=Int[], ξv=Float64[],
    ξh=Float64[],
    ξ_vumps=Float64[]
)

logfile = open("boundary_spectrum_diagnostics.txt", "w")


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
        ctm_alg = SequentialCTMRG(; maxiter=200, tol=1e-9, verbosity=2)

        env_Z, = leading_boundary(env_Z, Z, ctm_alg)
        ξv, ξh, λv, λh, λ_h_north, λ_h_south = correlation_length_check(Z, env_Z)
        ξ_vumps = correlation_length_VUMPS(Z, chi)
        # Append to results
        push!(results, (hx, hz, chi, D, ξv[1], ξh[1], ξ_vumps))

        # Log results
        println(logfile, "D=$D, χ=$chi, hx=$hx, hz=$hz")
        println(logfile, "horizontal spectrum: ", λh)
        println(logfile, "vertical spectrum: ", λv)
        println(logfile, "horizontal north spectrum: ", λ_h_north)
        println(logfile, "horizontal south spectrum: ", λ_h_south)

        println(logfile)  # empty line for separation
    catch e
        @warn "Skipping $file due to error" exception = e
    end

end
close(logfile)


CSV.write("correlationlengths.csv", results)


