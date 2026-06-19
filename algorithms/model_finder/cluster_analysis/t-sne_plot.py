def plot_tsne_clusters(
    X: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    perplexity: float = 30,
    random_state: int = 42,
    figsize: tuple[float, float] = (8, 6),
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = np.asarray(X, dtype=float)

    labels = np.asarray(labels)
    unique_labels = np.sort(np.unique(labels))
    display_map = {cluster_id: i + 1 for i, cluster_id in enumerate(unique_labels)}
    display_labels = np.array([display_map[label] for label in labels])

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
        learning_rate=200,
        max_iter=1000,
    )
    embedding = tsne.fit_transform(X_values)

    plot_df = pd.DataFrame(
        {
            "tSNE_1": embedding[:, 0],
            "tSNE_2": embedding[:, 1],
            "Cluster": display_labels,
        }
    )

    colors = ["#df6f85", "#a5aa00", "#14c7b5", "#8f84e8", "#f39c34", "#5dade2"]

    fig, ax = plt.subplots(figsize=figsize)

    for i, cluster_num in enumerate(sorted(plot_df["Cluster"].unique())):
        subset = plot_df[plot_df["Cluster"] == cluster_num]
        ax.scatter(
            subset["tSNE_1"],
            subset["tSNE_2"],
            s=42,
            c=colors[i % len(colors)],
            edgecolors="black",
            linewidths=0.6,
            alpha=1.0,
            label=str(cluster_num),
        )

    ax.set_title("t-SNE Cluster Plot", fontsize=16, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")

    legend = ax.legend(
        title="Cluster",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        handletextpad=0.4,
        labelspacing=0.8,
    )
    legend.get_title().set_fontsize(12)
    for text in legend.get_texts():
        text.set_fontsize(11)

    plt.tight_layout()
    plt.show()

    return plot_df


tsne_plot_data = plot_tsne_clusters(
    X=X_cluster,
    labels=best["labels"],
    perplexity=30,
    random_state=42,
    figsize=(8, 6),
)
