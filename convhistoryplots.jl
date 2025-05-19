using JLD2, FileIO, Glob, Plots, PDFmerger

gr()  # Use the GR backend for consistent PDF output

folder = "Saved_content"
files = glob("final_Psi_trivial_1e4*_hx=*_hz=*_χ=*_D=*.jld2", folder)

tmpdir = mktempdir()
plotpaths = String[]

function extract_params(filename)
    # Use regex to extract hx, hz, chi, D
    m = match(r"hx=([0-9.eE+-]+)_hz=([0-9.eE+-]+)_χ=([0-9]+)_D=([0-9]+)", filename)
    if m === nothing || any(x -> x === nothing, m.captures)
        error("Could not extract parameters from filename: $filename")
    end
    hx = parse(Float64, m.captures[1])
    hz = parse(Float64, m.captures[2])
    chi = parse(Int, m.captures[3])
    D = parse(Int, m.captures[4])
    return hx, hz, chi, D
end

for file in files
    try
        filename = basename(file)
        hx, hz, chi, D = extract_params(filename)

        f = jldopen(file, "r")
        conv = f["convhistory"]
        close(f)

        E = conv[:, 1]
        delE = conv[:, 2]
        iters = 1:length(E)

        label = "hx=$hx, hz=$hz, χ=$chi, D=$D"
        println("Processing $label")
        p1 = plot(iters, E, xlabel="Iteration", ylabel="E", label=label, title="Energy vs. Iteration")
        p2 = plot(iters, delE, xlabel="Iteration", ylabel="ΔE", label=label, title="ΔE vs. Iteration", yscale=:log10)

        combined = plot(p1, p2, layout=(1, 2), size=(1000, 400), title=label)

        outname = joinpath(tmpdir, filename * ".pdf")
        push!(plotpaths, outname)
        savefig(combined, outname)

    catch e
        @warn "Error processing $file" exception=e
    end
end

# Merge all into one PDF
merge_pdfs(plotpaths, "convergence_plots_combined.pdf")