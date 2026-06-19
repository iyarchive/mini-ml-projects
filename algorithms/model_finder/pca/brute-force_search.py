# PCA brute-force search

@dataclass
class PCASearchConfig:
    candidate_columns: list[str]
    subset_min_size: int = 8
    subset_max_size: int | None = None
    max_subsets: int | None = None

    min_components: int = 2
    max_components: int | None = 5
    rotations: tuple[str | None, ...] = ("promax", "oblimin", "varimax")

    kmo_min: float = 0.70
    item_kmo_min: float = 0.45
    bartlett_p_max: float = 0.05

    min_communality: float = 0.30
    primary_loading_min: float = 0.40
    secondary_loading_max: float = 0.35
    min_loading_gap: float = 0.15
    max_cross_loading_items: int = 2
    min_items_per_component: int = 2

    min_total_explained_variance: float = 0.50
    require_eigenvalue_over_one: bool = False

    top_n: int = 20
    verbose: bool = True


def report_p_value(p: float) -> str:
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def _generate_subsets(
    columns: list[str],
    min_size: int,
    max_size: int | None,
    max_subsets: int | None,
) -> list[tuple[str, ...]]:
    max_size = max_size or len(columns)
    subsets: list[tuple[str, ...]] = []

    for size in range(min_size, max_size + 1):
        for combo in combinations(columns, size):
            subsets.append(combo)
            if max_subsets is not None and len(subsets) >= max_subsets:
                return subsets

    return subsets


def _compute_pca_loadings(
    data: pd.DataFrame,
    n_components: int,
    rotation: str | None,
) -> tuple[pd.DataFrame, list[float], float]:
    corr = data.corr()
    eigenvalues, eigenvectors = np.linalg.eigh(corr.to_numpy())

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    retained_eigenvalues = eigenvalues[:n_components]
    retained_vectors = eigenvectors[:, :n_components]
    raw_loadings = retained_vectors * np.sqrt(retained_eigenvalues)

    if rotation is not None:
        rotated = Rotator(method=rotation).fit_transform(raw_loadings)
        loadings = pd.DataFrame(
            rotated,
            index=data.columns,
            columns=[f"PC{i + 1}" for i in range(n_components)],
        )
    else:
        loadings = pd.DataFrame(
            raw_loadings,
            index=data.columns,
            columns=[f"PC{i + 1}" for i in range(n_components)],
        )

    explained_variance = float(np.sum(retained_eigenvalues) / data.shape[1])
    return loadings, retained_eigenvalues.tolist(), explained_variance


def _primary_secondary(loadings: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    abs_loadings = loadings.abs().to_numpy()
    sorted_abs = np.sort(abs_loadings, axis=1)[:, ::-1]
    primary = pd.Series(sorted_abs[:, 0], index=loadings.index, name="primary_loading")

    if loadings.shape[1] > 1:
        secondary_vals = sorted_abs[:, 1]
    else:
        secondary_vals = np.zeros(loadings.shape[0])

    secondary = pd.Series(secondary_vals, index=loadings.index, name="secondary_loading")
    return primary, secondary


def _cross_loading_mask(
    loadings: pd.DataFrame,
    secondary_loading_max: float,
    min_loading_gap: float,
) -> pd.Series:
    primary, secondary = _primary_secondary(loadings)
    gap = primary - secondary
    return (secondary >= secondary_loading_max) | (gap < min_loading_gap)


def _items_per_component(
    loadings: pd.DataFrame,
    primary_loading_min: float,
) -> dict[str, int]:
    abs_loadings = loadings.abs()
    strongest_component = abs_loadings.idxmax(axis=1)
    strongest_value = abs_loadings.max(axis=1)
    valid = strongest_value >= primary_loading_min

    counts = {col: 0 for col in loadings.columns}
    for component in strongest_component[valid]:
        counts[str(component)] += 1
    return counts


def _highest_only_loadings(loadings: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    highest_only = pd.DataFrame(
        np.nan,
        index=loadings.index,
        columns=loadings.columns,
    )

    strongest_component = loadings.abs().idxmax(axis=1)

    for variable, component in strongest_component.items():
        highest_only.loc[variable, component] = round(loadings.loc[variable, component], decimals)

    return highest_only


def brute_force_pca_search(
    data: pd.DataFrame,
    config: PCASearchConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    candidate_data = data.loc[:, config.candidate_columns].copy()

    if candidate_data.isna().any().any():
        candidate_data = candidate_data.dropna(axis=0)

    bad_cols = [col for col in candidate_data.columns if candidate_data[col].nunique(dropna=True) <= 1]
    if bad_cols:
        raise ValueError(f"Remove constant columns first: {bad_cols}")

    subsets = _generate_subsets(
        columns=config.candidate_columns,
        min_size=config.subset_min_size,
        max_size=config.subset_max_size,
        max_subsets=config.max_subsets,
    )

    if config.verbose:
        print(f"Testing {len(subsets)} subsets from {len(config.candidate_columns)} candidate variables.")

    all_results: list[dict[str, Any]] = []

    for i, subset in enumerate(subsets, start=1):
        subset_df = candidate_data.loc[:, list(subset)]

        try:
            kmo_per_variable, kmo_overall = calculate_kmo(subset_df)
            chi_square_value, bartlett_p = calculate_bartlett_sphericity(subset_df)
        except Exception:
            continue

        item_kmo = pd.Series(kmo_per_variable, index=subset_df.columns, name="item_kmo")

        if kmo_overall < config.kmo_min:
            continue

        if item_kmo.min() < config.item_kmo_min:
            continue

        if bartlett_p > config.bartlett_p_max:
            continue

        max_components = min(
            config.max_components or (subset_df.shape[1] - 1),
            subset_df.shape[1] - 1,
        )
        min_components = min(config.min_components, max_components)

        if config.verbose and (i == 1 or i % 25 == 0 or i == len(subsets)):
            print(
                f"[{i}/{len(subsets)}] Current subset: {list(subset)} | "
                f"KMO={kmo_overall:.3f} | Bartlett {report_p_value(bartlett_p)}"
            )

        for n_components in range(min_components, max_components + 1):
            for rotation in config.rotations:
                try:
                    loadings, eigenvalues_retained, explained_variance = _compute_pca_loadings(
                        data=subset_df,
                        n_components=n_components,
                        rotation=rotation,
                    )
                except Exception:
                    continue

                communalities = (loadings**2).sum(axis=1).rename("communality")
                primary, secondary = _primary_secondary(loadings)
                cross_mask = _cross_loading_mask(
                    loadings=loadings,
                    secondary_loading_max=config.secondary_loading_max,
                    min_loading_gap=config.min_loading_gap,
                )
                cross_loading_items = loadings.index[cross_mask].tolist()
                items_per_component = _items_per_component(
                    loadings=loadings,
                    primary_loading_min=config.primary_loading_min,
                )
                highest_only_loadings = _highest_only_loadings(loadings.round(4))

                if config.require_eigenvalue_over_one and any(ev <= 1.0 for ev in eigenvalues_retained):
                    continue

                if explained_variance < config.min_total_explained_variance:
                    continue

                if communalities.min() < config.min_communality:
                    continue

                if primary.min() < config.primary_loading_min:
                    continue

                if int(cross_mask.sum()) > config.max_cross_loading_items:
                    continue

                if any(v < config.min_items_per_component for v in items_per_component.values()):
                    continue

                score = (
                    (float(kmo_overall) * 100.0)
                    + (explained_variance * 50.0)
                    + (float(communalities.mean()) * 20.0)
                    + (float(primary.mean()) * 20.0)
                    - (int(cross_mask.sum()) * 15.0)
                )

                all_results.append(
                    {
                        "variables": list(subset),
                        "n_variables": len(subset),
                        "n_components": n_components,
                        "rotation": rotation,
                        "kmo_overall": float(kmo_overall),
                        "min_item_kmo": float(item_kmo.min()),
                        "bartlett_p": float(bartlett_p),
                        "bartlett_report": report_p_value(float(bartlett_p)),
                        "chi_square_value": float(chi_square_value),
                        "explained_variance_ratio_sum": float(explained_variance),
                        "min_communality": float(communalities.min()),
                        "mean_communality": float(communalities.mean()),
                        "min_primary_loading": float(primary.min()),
                        "mean_primary_loading": float(primary.mean()),
                        "n_cross_loading_items": int(cross_mask.sum()),
                        "cross_loading_items": cross_loading_items,
                        "items_per_component": items_per_component,
                        "eigenvalues_retained": [float(v) for v in eigenvalues_retained],
                        "score": float(score),
                        "loadings": loadings.round(4),
                        "highest_only_loadings": highest_only_loadings,
                        "communalities": communalities.round(4),
                        "item_kmo": item_kmo.round(4),
                    }
                )

    all_results = sorted(
        all_results,
        key=lambda r: (
            -r["score"],
            -r["kmo_overall"],
            -r["explained_variance_ratio_sum"],
            -r["mean_communality"],
            -r["mean_primary_loading"],
            r["n_cross_loading_items"],
        ),
    )

    top_results = all_results[: config.top_n]

    summary_rows = []
    for result in top_results:
        summary_rows.append(
            {
                "variables": ", ".join(result["variables"]),
                "n_variables": result["n_variables"],
                "n_components": result["n_components"],
                "rotation": result["rotation"],
                "kmo_overall": result["kmo_overall"],
                "min_item_kmo": result["min_item_kmo"],
                "bartlett_p": result["bartlett_report"],
                "chi_square_value": result["chi_square_value"],
                "explained_variance_ratio_sum": result["explained_variance_ratio_sum"],
                "min_communality": result["min_communality"],
                "mean_communality": result["mean_communality"],
                "min_primary_loading": result["min_primary_loading"],
                "mean_primary_loading": result["mean_primary_loading"],
                "n_cross_loading_items": result["n_cross_loading_items"],
                "items_per_component": result["items_per_component"],
                "eigenvalues_retained": result["eigenvalues_retained"],
                "score": result["score"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return summary_df, top_results

config = PCASearchConfig(
    candidate_columns=candidate_columns,
    subset_min_size=7,
    subset_max_size=8,
    max_subsets=None,
    min_components=2,
    max_components=2,
    rotations=("oblimin", "varimax"),
    kmo_min=0.80,
    item_kmo_min=0.45,
    bartlett_p_max=0.05,
    min_communality=0.30,
    primary_loading_min=0.40,
    secondary_loading_max=0.35,
    min_loading_gap=0.15,
    max_cross_loading_items=2,
    min_items_per_component=2,
    min_total_explained_variance=0.70,
    require_eigenvalue_over_one=False,
    top_n=20,
    verbose=True,
)

pca_search_summary, pca_search_results = brute_force_pca_search(X_scaled, config)

if pca_search_results:
    best_model = pca_search_results[0]

    print("\nBest model summary:")
    print(f"Variables: {best_model['variables']}")
    print(f"Components: {best_model['n_components']}")
    print(f"Rotation: {best_model['rotation']}")
    print(f"Overall KMO: {best_model['kmo_overall']:.3f}")
    print(f"Minimum item KMO: {best_model['min_item_kmo']:.3f}")
    print(f"Bartlett's test: chi-square = {best_model['chi_square_value']:.3f}, {best_model['bartlett_report']}")
    print(f"Explained variance: {best_model['explained_variance_ratio_sum']:.3f}")
    print(f"Minimum communality: {best_model['min_communality']:.3f}")
    print(f"Mean communality: {best_model['mean_communality']:.3f}")
    print(f"Minimum primary loading: {best_model['min_primary_loading']:.3f}")
    print(f"Mean primary loading: {best_model['mean_primary_loading']:.3f}")
    print(f"Cross-loading items: {best_model['n_cross_loading_items']}")
    print(f"Items per component: {best_model['items_per_component']}")
    print(f"Eigenvalues retained: {[round(v, 3) for v in best_model['eigenvalues_retained']]}")

    print("\nBest model item KMO:")
    display(best_model["item_kmo"].to_frame())

    print("\nBest model communalities:")
    display(best_model["communalities"].to_frame())

    print("\nBest model highest loadings only:")
    display(best_model["highest_only_loadings"].fillna(""))

else:
    print("No PCA model passed the thresholds. Loosen the rules or reduce the candidate set.")
