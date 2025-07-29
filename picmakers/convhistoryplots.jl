using JLD2, FileIO, Glob, Plots, PDFmerger, DataStructures

gr()  # Use the GR backend for consistent PDF output

folder = "Saved_content_sequential/"
files = glob("final_Psi_trivial_1e4*_hx=*_hz=*_χ=*_D=*.jld2", folder)

tmpdir = mktempdir()
plotpaths = String[]
default(
    fontfamily="Computer Modern",
    grid=:dash,
    framestyle=:box,
    size=(600, 350),
    legend=:topright,
    linewidth=2,
    markerstrokewidth=0.5
)

function extract_params(filename)
    m = match(r"hx=([0-9.eE+-]+)_hz=([0-9.eE+-]+)_χ=([0-9]+)_D=([0-9]+)", filename)
    if m === nothing || any(x -> x === nothing, m.captures)
        return nothing
    end
    hx = parse(Float64, m.captures[1])
    hz = parse(Float64, m.captures[2])
    chi = parse(Int, m.captures[3])
    D = parse(Int, m.captures[4])
    return (D, chi, hx, hz)
end

# Sort files by extracted parameters
file_map = OrderedDict{String,Tuple{Int,Int,Float64,Float64}}()
for file in files
    p = extract_params(basename(file))
    if p !== nothing
        file_map[file] = p
    end
end

sorted_files = sort(collect(file_map), by=x -> x[2])  # Sort by (D, chi, hx, hz)

for (file, (D, chi, hx, hz)) in sorted_files
    try
        filename = basename(file)
        D, chi, hx, hz = extract_params(filename)

        f = jldopen(file, "r")
        conv = f["convhistory"]
        close(f)

        E = conv[:, 1]
        delE = conv[:, 2]
        iters = 1:length(E)

        label = "hx=$hx, hz=$hz, χ=$chi, D=$D"
        println("Processing $label")


        p1 = plot(iters, E, xlabel="Iteration", ylabel="E", label="", title="Energy vs. Iteration", marker=:circle, markersize=4, color=:blue)
        plot!(iters, E, lw=1.5, label="E", color=:blue)
        p2 = plot(iters, delE, xlabel="Iteration", ylabel="ΔE", label="", title="ΔE vs. Iteration", yscale=:log10, marker=:circle, markersize=4, color=:red)
        plot!(iters, delE, lw=1.5, label="ΔE", color=:red)
        combined = plot(p1, p2, layout=(1, 2), size=(600, 350), title=label)

        outname = joinpath(tmpdir, filename * ".pdf")
        push!(plotpaths, outname)
        savefig(combined, outname)

    catch e
        @warn "Error processing $file" exception = e
    end
end

# Merge all into one PDF
merge_pdfs(plotpaths, "pics/convergence_plots_combined_sequential.pdf")