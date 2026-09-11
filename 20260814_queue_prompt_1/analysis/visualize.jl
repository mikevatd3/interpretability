using GLMakie
# Plotting posterior distributions with GLMakie --------------------------------

function plot_experiment_results(feature_names, chain)

    betas = chain[:β].data |> vec
    betas = hcat(betas...)'

    fig = Figure(size=(1200, 900))

    Label(fig[1, 1:2], "Posterior Distributions of Coefficients", fontsize=16, font=:bold)

    for (i, name) in enumerate(feature_names)
        row, col = fldmod1(i, 2)
        ax = Axis(fig[row + 1, col], title=name, xlabel="Coefficient value", ylabel="Density")

        col_data = betas[:, i]

        # Plot histogram
        hist!(ax, col_data, bins=30, color=(:steelblue, 0.7), normalization=:density)

        # Plot vertical lines for mean and credible interval
        mean_val = mean(col_data)
        ci_lower = quantile(col_data, 0.025)
        ci_upper = quantile(col_data, 0.975)

        vlines!(ax, mean_val, color=:red, linewidth=2, label="Mean")
        vlines!(ax, [ci_lower, ci_upper], color=:orange, linewidth=1.5, linestyle=:dash, label="95% CI")

        # axislegend draws inside the axis's own plot area instead of claiming a
        # grid cell -- putting the Legend in fig[row, col] alongside the Axis
        # (the old approach) makes Makie share that cell between the two,
        # collapsing the column width to the Legend's narrow content and
        # leaving the rest of the figure blank.
        axislegend(ax, position=:rt, labelsize=10)
    end

    # Explicit sizing so each part of the figure claims a known, fixed share of
    # the canvas: the title is a thin fixed band, and the 2x2 axes grid below it
    # splits the rest into equal halves both ways. Relative (not Auto) sizing is
    # what forces these rows/columns to actually expand to fill the figure
    # instead of shrinking to fit their content.
    rowsize!(fig.layout, 1, Relative(0.06))
    rowsize!(fig.layout, 2, Relative(0.47))
    rowsize!(fig.layout, 3, Relative(0.47))
    colsize!(fig.layout, 1, Relative(0.5))
    colsize!(fig.layout, 2, Relative(0.5))

    save("posterior_distributions.png", fig)
    println("\nPlot saved to posterior_distributions.png")
    return fig
end

