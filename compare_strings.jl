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
include("new_toolbox.jl")

using JLD2, FileIO, Glob, CSV, DataFrames

# Path to folder
folder = "Saved_content"
files = glob("final_Psi_trivial_1e4*_hx=*_hz=*_χ=*_D=*.jld2", folder)

results = DataFrame(
    hx=Float64[],
    hz=Float64[],
    chi=Int[],
    D=Int[],
    E=[], infinite_tHooft=ComplexF64[],
    infinite_Wilson=ComplexF64[], infinite_Wilson_odd=ComplexF64[],
    FMstring=Float64[], PKstring_odd=Float64[]
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
        env_Z, = leading_boundary(env_Z, Z, SequentialCTMRG(; maxiter=200, tol=1e-9, verbosity=2))

        env = retrieve_old_environment(env_Z, Ψ)
        vals_tHooft_trivial, vals_tHooft, vals_Wilson_trivial, vals_Wilson_trivial_odd, vals_Wilson, vecs_tHooft_trivial, vecs_tHooft, vecs_Wilson_trivial, vecs_wilson_trivial_odd, vecs_Wilson = strings_CTMRG(Ψ, env)
        infinite_tHooft = vals_tHooft[1] / vals_tHooft_trivial[1]
        infinite_Wilson = vals_Wilson[1] / vals_Wilson_trivial[1]
        infinite_Wilson_odd = vals_Wilson[1] / vals_Wilson_trivial_odd[1]
        FMstring = abs(dot(vecs_tHooft_trivial[1], vecs_tHooft[1]))
        PKstring_odd = abs(dot(vecs_wilson_trivial_odd[1], vecs_Wilson[1]))
        @show infinite_tHooft
        # Append to results
        push!(results, (hx, hz, chi, D, E, infinite_tHooft, infinite_Wilson, infinite_Wilson_odd, FMstring, PKstring_odd))

    catch e
        @warn "Skipping $file due to error" exception = e
    end
end

# Save to CSV
CSV.write("strings.csv", results)