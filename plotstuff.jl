using CSV, DataFrames, Plots

# Load the data
df = CSV.read("results.csv", DataFrame)
D = 6
chi = 36
# Filter for D = 4, chi = 36
df_filtered = filter(row -> row.D == D && row.chi == chi, df)
# Get unique grid values
hx_vals = sort(unique(df_filtered.hx))
hz_vals = sort(unique(df_filtered.hz))

function get_infinite_wilson(hx, hz, df)
    rows = filter(row -> row.hx == hx && row.hz == hz, df)
    if !isempty(rows)
        val_str = rows.infinite_Wilson[1]
        val = parse(ComplexF64, val_str)
        return abs(real(val))
    else
        return NaN
    end
end

function get_Energy(hx, hz, df)
    rows = filter(row -> row.hx == hx && row.hz == hz, df)
    if !isempty(rows)
        val_str = rows.E[1]
        val = val_str
        return val
    else
        return NaN
    end
end
function get_infinite_tHooft(hx, hz, df)
    rows = filter(row -> row.hx == hx && row.hz == hz, df)
    if !isempty(rows)
        val_str = rows.infinite_tHooft[1]
        val = parse(ComplexF64, val_str)
        return abs(real(val))
    else
        return NaN
    end
end
Z_energy = [get_Energy(hx, hz, df_filtered) for hz in hz_vals, hx in hx_vals]
Z_tHooft = [get_infinite_tHooft(hx, hz, df_filtered) for hz in hz_vals, hx in hx_vals]
Z_wilson = [get_infinite_wilson(hx, hz, df_filtered) for hz in hz_vals, hx in hx_vals]
colors_energy = vec(Z_energy)
colors_tHooft = vec(Z_tHooft)
colors_wilson = vec(Z_wilson)

xs = repeat(hx_vals, inner=length(hz_vals))
ys = repeat(hz_vals, outer=length(hx_vals))

scatter(xs, ys;
    zcolor = colors_energy,
    xlabel = "hx",
    ylabel = "hz",
    title = "Energy (D=$(D), χ=$(chi))",
    colorbar_title = "Energy",
    markersize = 8,
    markerstrokewidth = 0,
    marker = :circle
)

savefig("pics/GroundstateEnergy_D=$(D)_chi=$(chi).svg")

scatter(xs, ys;
    zcolor = colors_tHooft,
    xlabel = "hx",
    ylabel = "hz",
    title = "tHooft strings (D=$(D), χ=$(chi))",
    colorbar_title = "tHooft strings",
    markersize = 8,
    markerstrokewidth = 0,
    marker = :circle
)

savefig("pics/infinite_tHooft_D=$(D)_chi=$(chi).svg")

scatter(xs, ys;
    zcolor = colors_wilson,
    xlabel = "hx",
    ylabel = "hz",
    title = "Wilson strings (D=$(D), χ=$(chi))",
    colorbar_title = "Wilson strings",
    markersize = 8,
    markerstrokewidth = 0,
    marker = :circle
)

savefig("pics/infinite_Wilson_D=$(D)_chi=$(chi).svg")

