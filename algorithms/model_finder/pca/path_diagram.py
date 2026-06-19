def plot_best_model_path_diagram(
    best_model: dict,
    loading_threshold: float = 0.15,
    figsize: tuple[float, float] = (5, 8),
    title: str = "Rotated Component Path Diagram",
) -> None:
    loadings = best_model["loadings"].copy()

    if loadings.empty:
        raise ValueError("best_model['loadings'] is empty.")

    loadings.columns = [f"RC{i + 1}" for i in range(loadings.shape[1])]

    if "uniqueness" in best_model:
        uniqueness = pd.Series(best_model["uniqueness"]).reindex(loadings.index)
    else:
        uniqueness = 1 - (loadings**2).sum(axis=1)
        uniqueness.name = "uniqueness"

    n_factors = loadings.shape[1]
    n_items = loadings.shape[0]

    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    factor_x = 0.14
    item_x = 0.63
    factor_radius = 0.035
    box_width = 0.28
    box_height = min(0.06, 0.82 / max(n_items, 1))

    factor_fontsize = 10
    item_fontsize = 8
    title_fontsize = 11

    factor_y = np.linspace(0.76, 0.24, n_factors)
    item_y = np.linspace(0.90, 0.10, n_items)

    factor_pos = {col: (factor_x, y) for col, y in zip(loadings.columns, factor_y)}
    item_pos = {row: (item_x, y) for row, y in zip(loadings.index, item_y)}

    for factor_name, (x, y) in factor_pos.items():
        ax.add_patch(
            Circle(
                (x, y),
                factor_radius,
                facecolor="#f2f2f2",
                edgecolor="black",
                lw=0.8,
            )
        )
        ax.text(x, y, factor_name, ha="center", va="center", fontsize=factor_fontsize)

    for item_name, (x, y) in item_pos.items():
        ax.add_patch(
            Rectangle(
                (x - box_width / 2, y - box_height / 2),
                box_width,
                box_height,
                facecolor="#f7f7f7",
                edgecolor="#777777",
                lw=0.7,
            )
        )
        ax.text(x, y, str(item_name), ha="center", va="center", fontsize=item_fontsize)

        if pd.notna(uniqueness.get(item_name, np.nan)):
            ax.add_patch(
                FancyArrowPatch(
                    (0.96, y),
                    (x + box_width / 2, y),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    lw=0.8,
                    color="#4caf50",
                    alpha=0.6,
                )
            )

    max_abs = float(np.abs(loadings.values).max())
    max_abs = max(max_abs, 1e-9)

    for item_name in loadings.index:
        for factor_name in loadings.columns:
            value = float(loadings.loc[item_name, factor_name])

            if math.isnan(value) or abs(value) < loading_threshold:
                continue

            fx, fy = factor_pos[factor_name]
            ix, iy = item_pos[item_name]

            ax.add_patch(
                FancyArrowPatch(
                    (fx + factor_radius * 0.95, fy),
                    (ix - box_width / 2, iy),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    lw=0.4 + 2.8 * (abs(value) / max_abs),
                    color="darkgreen" if value >= 0 else "#d66a6a",
                    alpha=0.2 + 0.7 * (abs(value) / max_abs),
                    connectionstyle="arc3,rad=0.0",
                    shrinkA=0,
                    shrinkB=0,
                )
            )

    ax.set_title(title, fontsize=title_fontsize, pad=8)
    plt.tight_layout()
    plt.show()

plot_best_model_path_diagram(
    best_model=best_model,
    loading_threshold=0.15,
    figsize=(6, 4),
    title=f"Path Diagram ({best_model['rotation']}, {best_model['n_components']} components)",
)
