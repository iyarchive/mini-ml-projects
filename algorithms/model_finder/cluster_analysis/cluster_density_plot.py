def prepare_cluster_mean_plot_data(
    X: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X = np.asarray(X, dtype=float)
        X_df = pd.DataFrame(X, columns=[f"PC_{i + 1}" for i in range(X.shape[1])])

    labels = np.asarray(labels)
    unique_labels = np.sort(np.unique(labels))
    cluster_map = {cluster_id: i + 1 for i, cluster_id in enumerate(unique_labels)}
    display_labels = np.array([cluster_map[label] for label in labels])

    plot_df = X_df.copy()
    plot_df["Cluster"] = display_labels

    means = plot_df.groupby("Cluster").mean()
    ses = plot_df.groupby("Cluster").sem()

    return means, ses


def plot_cluster_mean_bars(
    X: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    figsize: tuple[float, float] = (8, 6),
) -> None:
    means, ses = prepare_cluster_mean_plot_data(X, labels)

    feature_names = list(means.columns)
    cluster_ids = list(means.index)
    n_features = len(feature_names)
    n_clusters = len(cluster_ids)

    colors = {
        1: "#df6f85",
        2: "#a5aa00",
        3: "#14c7b5",
        4: "#8f84e8",
        5: "#f39c34",
        6: "#5dade2",
    }

    x = np.arange(n_features)
    total_width = 0.78
    bar_width = total_width / n_clusters
    offsets = np.linspace(
        -total_width / 2 + bar_width / 2,
        total_width / 2 - bar_width / 2,
        n_clusters,
    )

    fig, ax = plt.subplots(figsize=figsize)

    for i, cluster_id in enumerate(cluster_ids):
        y = means.loc[cluster_id, feature_names].values
        yerr = ses.loc[cluster_id, feature_names].values

        ax.bar(
            x + offsets[i],
            y,
            width=bar_width,
            color=colors.get(cluster_id, "#999999"),
            edgecolor="black",
            linewidth=1.1,
            yerr=yerr,
            capsize=2.5,
            error_kw={"elinewidth": 1.8, "capthick": 1.8},
            label=str(cluster_id),
        )

    ax.set_title("Cluster Mean Plots", fontsize=16, fontweight="bold", loc="left", pad=14)

    ax.set_ylabel("Cluster Mean", fontsize=12)
    ax.set_xlabel("")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=20, ha="right", fontsize=11)

    y_min = min(-1, float(np.floor((means - ses).min().min())))
    y_max = max(1, float(np.ceil((means + ses).max().max())))
    ax.set_ylim(y_min, y_max)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b0b0b0")
    ax.spines["bottom"].set_color("#b0b0b0")
    ax.tick_params(axis="both", length=8, width=1, color="#888888", labelsize=11)

    legend_handles = [
        Patch(
            facecolor=colors.get(cluster_id, "#999999"),
            edgecolor="black",
            label=str(cluster_id),
        )
        for cluster_id in cluster_ids
    ]

    legend = ax.legend(
        handles=legend_handles,
        title="Cluster",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.92),
        borderaxespad=0,
        handlelength=1.6,
        handleheight=1.6,
        handletextpad=0.5,
        labelspacing=0.35,
        fontsize=11,
        title_fontsize=12,
    )

    plt.tight_layout()
    plt.show()


plot_cluster_mean_bars(
    X=X_cluster,
    labels=best["labels"],
    figsize=(8, 6),
)

def plot_cluster_density_ridges(
    X: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    figsize: tuple[float, float] = (8, 7),
    title: str = "Cluster Density Plots",
    subtitle: str = "All Features",
) -> None:
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X = np.asarray(X, dtype=float)
        X_df = pd.DataFrame(X, columns=[f"PC_{i + 1}" for i in range(X.shape[1])])

    labels = np.asarray(labels)
    unique_labels = np.sort(np.unique(labels))
    cluster_map = {cluster_id: i + 1 for i, cluster_id in enumerate(unique_labels)}
    display_labels = np.array([cluster_map[label] for label in labels])

    plot_df = X_df.copy()
    plot_df["Cluster"] = display_labels

    feature_names = list(X_df.columns)
    n_features = len(feature_names)

    colors = {
        1: "#df9aa9",
        2: "#b8bc63",
        3: "#62c8bd",
        4: "#a79ae3",
        5: "#f3b562",
        6: "#7cb5ec",
    }

    global_min = float(X_df.min().min())
    global_max = float(X_df.max().max())
    x_pad = 0.15 * (global_max - global_min if global_max > global_min else 1.0)
    x_grid = np.linspace(global_min - x_pad, global_max + x_pad, 500)

    fig, ax = plt.subplots(figsize=figsize)

    y_positions = np.arange(n_features, 0, -1)
    ridge_height = 0.72

    for y_pos, feature in zip(y_positions, feature_names):
        ax.hlines(y=y_pos, xmin=x_grid.min(), xmax=x_grid.max(), color="black", linewidth=1.1, zorder=1)

        for cluster_num in sorted(np.unique(display_labels)):
            values = plot_df.loc[plot_df["Cluster"] == cluster_num, feature].dropna().to_numpy()

            if len(values) == 0:
                continue

            if len(np.unique(values)) < 2 or np.std(values) < 1e-8:
                mu = float(np.mean(values))
                sigma = 0.08 if np.std(values) < 1e-8 else float(np.std(values))
                density = np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            else:
                kde = gaussian_kde(values)
                density = kde(x_grid)

            density = density / density.max() * ridge_height

            ax.fill_between(
                x_grid,
                y_pos,
                y_pos + density,
                color=colors.get(cluster_num, "#999999"),
                alpha=0.75,
                edgecolor="black",
                linewidth=1.1,
                zorder=2,
            )

    ax.set_title(title, fontsize=16, fontweight="bold", loc="left", pad=14)
    ax.text(
        0.01,
        0.93,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    ax.set_xlabel("Value", fontsize=12, labelpad=10)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(feature_names, fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")

    ax.tick_params(axis="x", length=6, color="#666666")
    ax.tick_params(axis="y", length=0)

    legend_handles = [
        Patch(facecolor=colors.get(cluster_num, "#999999"), edgecolor="black", label=str(cluster_num))
        for cluster_num in sorted(np.unique(display_labels))
    ]

    ax.legend(
        handles=legend_handles,
        title="Cluster",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 0.86),
        borderaxespad=0,
        fontsize=11,
        title_fontsize=12,
        handlelength=1.6,
        handleheight=1.6,
        labelspacing=0.35,
    )

    plt.tight_layout()
    plt.show()


plot_cluster_density_ridges(
    X=X_cluster,
    labels=best["labels"],
    figsize=(8, 7),
)

cluster_share_table = (
    df["Cluster"]
    .value_counts(normalize=True)
    .sort_index()
    .rename_axis("Cluster")
    .reset_index(name="Proportion")
)

cluster_share_table["Cluster"] = range(1, len(cluster_share_table) + 1)

cluster_share_table
