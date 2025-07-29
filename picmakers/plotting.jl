using CSV, DataFrames, Plots

default(fontfamily = "Computer Modern", size=(700, 550), legend=false, grid=:dash)

# Load the data
df = CSV.read("strings.csv", DataFrame)

# Get all unique D and chi values
Ds = unique(df.D)
chis = unique(df.chi)

# Ensure output folder exists
mkpath("pics")

# Loop over combinations
for D in Ds, chi in chis
    df_filtered = filter(row -> row.D == D && row.chi == chi, df)

    if nrow(df_filtered) == 0
        @info "Skipping D=$D, chi=$chi — no data"
        continue
    end

    try
        # Parse values
        hx = df_filtered.hx
        hz = df_filtered.hz
        tHooft_vals = abs.(real.(parse.(ComplexF64, df_filtered.infinite_tHooft)))
        Wilson_vals = abs.(real.(parse.(ComplexF64, df_filtered.infinite_Wilson)))
        FMstring_vals = df_filtered.FMstring
        PKstring_odd = df_filtered.PKstring_odd
        # mv = 1 ./ df_filtered.ξv
        # mh = 1 ./ df_filtered.ξh
        # E_vals = df_filtered.E

        function plot_and_save(zvals, title_str, cbar_str, filename, cmap)
            scatter(hx, hz;
                zcolor = zvals,
                colorbar = true,
                colorbar_title = cbar_str,
                xlabel = "hx",
                ylabel = "hz",
                title = "$title_str (D=$D, χ=$chi)",
                markersize = 10,
                markerstrokewidth = 0.5,
                markerstrokealpha = 0.3,
                marker = :circle,
                c = cmap,
                tickfontsize = 12,
                guidefontsize = 14,
                titlefontsize = 16,
            )
            savefig("pics/$filename")
        end

        plot_and_save(tHooft_vals,
            "Infinite t'Hooft String",
            "Infinite t'Hooft",
            "infinite_tHooft_D=$(D)_chi=$(chi).svg",
            :magma)

        plot_and_save(Wilson_vals,
            "Infinite Wilson String",
            "Infinite Wilson",
            "infinite_Wilson_D=$(D)_chi=$(chi).svg",
            :magma)
        plot_and_save(FMstring_vals,
            "Fredenhagen-Marcu String",
            "Fredenhagen-Marcu",
            "FMstring_D=$(D)_chi=$(chi).svg",
            :magma)

        plot_and_save(PKstring_odd,
            "Pfeifer-Kogut String (odd)",
            "Pfeifer-Kogut",
            "PKstring_odd_D=$(D)_chi=$(chi).svg",
            :magma)

        # plot_and_save(E_vals,
        #     "Ground State Energy",
        #     "Energy",
        #     "GroundstateEnergy_D=$(D)_chi=$(chi).svg",
        #     :coolwarm)

        # plot_and_save(mv,
        #     "Inverse Correlation Length (v)",
        #     "1/ξv",
        #     "InverseCorrelationLength_v_D=$(D)_chi=$(chi).svg",
        #     :plasma)

        # plot_and_save(mh,
        #     "Inverse Correlation Length (h)",
        #     "1/ξₕ",
        #     "InverseCorrelationLength_h_D=$(D)_chi=$(chi).svg",
        #     :cividis)

        @info "Plotted D=$D, chi=$chi"

    catch e
        @warn "Failed for D=$D, chi=$chi" exception = e
    end
end