def compute_cluster_fit_table(X: pd.DataFrame | np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    n, p = X.shape
    k = len(unique_labels)

    grand_mean = X.mean(axis=0)
    tss = float(((X - grand_mean) ** 2).sum())

    wss = 0.0
    for cluster_id in unique_labels:
        cluster_points = X[labels == cluster_id]
        cluster_mean = cluster_points.mean(axis=0)
        wss += float(((cluster_points - cluster_mean) ** 2).sum())

    bss = tss - wss
    r2 = bss / tss if tss > 0 else np.nan

    sil = silhouette_score(X, labels)

    # Pseudo log-likelihood for hard clustering under spherical Gaussian error.
    sigma2 = max(wss / (n * p), 1e-12)
    log_likelihood = -0.5 * n * p * (np.log(2 * np.pi * sigma2) + 1)

    # Parameters: centroids + mixing proportions + shared variance
    n_params = k * p + (k - 1) + 1
    aic = 2 * n_params - 2 * log_likelihood
    bic = n_params * np.log(n) - 2 * log_likelihood

    return pd.DataFrame(
        [
            {
                "Clusters": int(k),
                "N": int(n),
                "R²": float(r2),
                "AIC": float(aic),
                "BIC": float(bic),
                "Silhouette": float(sil),
            }
        ]
    )


cluster_fit_table = compute_cluster_fit_table(X_cluster, best["labels"])

styled_cluster_fit_table = (
    cluster_fit_table.style
    .hide(axis="index")
    .format(
        {
            "Clusters": "{:.0f}",
            "N": "{:.0f}",
            "R²": "{:.3f}",
            "AIC": "{:.0f}",
            "BIC": "{:.0f}",
            "Silhouette": "{:.3f}",
        }
    )
    .set_table_styles(
        [
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("margin", "0"),
                    ("font-family", "Arial, sans-serif"),
                    ("font-size", "18px"),
                    ("border-top", "3px solid black"),
                    ("border-bottom", "3px solid black"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("font-weight", "normal"),
                    ("text-align", "center"),
                    ("padding", "8px 18px"),
                    ("border-bottom", "2px solid black"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("text-align", "center"),
                    ("padding", "10px 18px"),
                ],
            },
        ]
    )
)

display(styled_cluster_fit_table)

def compute_cluster_detail_table(
    X: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    unique_labels = np.sort(np.unique(labels))
    sil_samples = silhouette_samples(X, labels)

    cluster_sizes = {}
    cluster_wss = {}
    cluster_silhouette = {}

    for cluster_id in unique_labels:
        mask = labels == cluster_id
        X_k = X[mask]
        centroid_k = X_k.mean(axis=0)

        cluster_sizes[int(cluster_id)] = int(mask.sum())
        cluster_wss[int(cluster_id)] = float(((X_k - centroid_k) ** 2).sum())
        cluster_silhouette[int(cluster_id)] = float(sil_samples[mask].mean())

    total_wss = sum(cluster_wss.values())
    cluster_heterogeneity_prop = {
        k: (v / total_wss if total_wss > 0 else np.nan)
        for k, v in cluster_wss.items()
    }

    display_labels = [str(i + 1) for i in range(len(unique_labels))]
    label_map = {disp: int(cluster_id) for disp, cluster_id in zip(display_labels, unique_labels)}

    table = pd.DataFrame(
        {
            disp: {
                "Size": cluster_sizes[label_map[disp]],
                "Explained proportion\nwithin-cluster\nheterogeneity": cluster_heterogeneity_prop[label_map[disp]],
                "Within sum of\nsquares": cluster_wss[label_map[disp]],
                "Silhouette\nscore": cluster_silhouette[label_map[disp]],
            }
            for disp in display_labels
        }
    )

    table.index.name = "Cluster"
    return table


def format_cluster_detail_table_for_display(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy().astype(object)

    for row_name in formatted.index:
        if row_name == "Size":
            formatted.loc[row_name] = formatted.loc[row_name].map(lambda x: f"{x:.0f}")
        elif row_name == "Within sum of\nsquares":
            formatted.loc[row_name] = formatted.loc[row_name].map(lambda x: f"{x:.1f}")
        else:
            formatted.loc[row_name] = formatted.loc[row_name].map(lambda x: f"{x:.3f}")

    return formatted


cluster_detail_table = compute_cluster_detail_table(X_cluster, best["labels"])
cluster_detail_table_display = format_cluster_detail_table_for_display(cluster_detail_table)

styled_cluster_detail_table = (
    cluster_detail_table_display.style
    .set_table_styles(
        [
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("font-family", "Arial, sans-serif"),
                    ("font-size", "16px"),
                    ("border-top", "2px solid black"),
                    ("border-bottom", "2px solid black"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "normal"),
                    ("padding", "6px 14px"),
                    ("border-bottom", "2px solid black"),
                ],
            },
            {
                "selector": "tbody th",
                "props": [
                    ("text-align", "left"),
                    ("font-weight", "normal"),
                    ("padding", "6px 14px"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("text-align", "center"),
                    ("padding", "6px 14px"),
                ],
            },
        ]
    )
)

display(styled_cluster_detail_table)

def infer_distance_metric(best_model: pd.Series | dict) -> str:
    if isinstance(best_model, pd.Series):
        best_model = best_model.to_dict()

    if "metric" in best_model and pd.notna(best_model["metric"]):
        return str(best_model["metric"])
    if "affinity" in best_model and pd.notna(best_model["affinity"]):
        return str(best_model["affinity"])

    algorithm = str(best_model.get("algorithm", "")).lower()

    # In your notebook:
    # - KMeans uses Euclidean distance
    # - AgglomerativeClustering was created without metric/affinity,
    #   so sklearn defaulted to Euclidean for the fitted model
    if algorithm in {"kmeans", "ward", "complete", "average"}:
        return "euclidean"

    return "euclidean"


def pearson_gamma_from_distances(dist_matrix: np.ndarray, labels: np.ndarray) -> float:
    n = len(labels)
    iu = np.triu_indices(n, k=1)

    d = dist_matrix[iu]
    between = (labels[iu[0]] != labels[iu[1]]).astype(float)

    if np.std(d) == 0 or np.std(between) == 0:
        return np.nan

    return float(np.corrcoef(d, between)[0, 1])


def compute_model_performance_metrics(
    X: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
    metric: str,
) -> pd.DataFrame:
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    unique_labels = np.sort(np.unique(labels))

    dist_matrix = pairwise_distances(X, metric=metric)

    max_diameter = 0.0
    min_separation = np.inf

    for cluster_id in unique_labels:
        idx = np.where(labels == cluster_id)[0]
        if len(idx) >= 2:
            within = dist_matrix[np.ix_(idx, idx)]
            diameter = float(np.max(within))
            max_diameter = max(max_diameter, diameter)

    for i, cluster_i in enumerate(unique_labels):
        idx_i = np.where(labels == cluster_i)[0]
        for cluster_j in unique_labels[i + 1:]:
            idx_j = np.where(labels == cluster_j)[0]
            between = dist_matrix[np.ix_(idx_i, idx_j)]
            separation = float(np.min(between))
            min_separation = min(min_separation, separation)

    dunn_index = (
        float(min_separation / max_diameter)
        if max_diameter > 0 and np.isfinite(min_separation)
        else np.nan
    )

    counts = np.array([(labels == cluster_id).sum() for cluster_id in unique_labels], dtype=float)
    proportions = counts / counts.sum()
    entropy = float(-(proportions * np.log(proportions)).sum())

    pearsons_gamma = pearson_gamma_from_distances(dist_matrix, labels)
    calinski = float(calinski_harabasz_score(X, labels))

    return pd.DataFrame(
        {
            "Value": [
                max_diameter,
                min_separation,
                pearsons_gamma,
                dunn_index,
                entropy,
                calinski,
            ]
        },
        index=[
            "Maximum\ndiameter",
            "Minimum\nseparation",
            "Pearson's\nγ",
            "Dunn index",
            "Entropy",
            "Calinski-\nHarabasz\nindex",
        ],
    )


def format_model_performance_table_for_display(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy().astype(object)

    for row_name in formatted.index:
        value = table.loc[row_name, "Value"]

        if row_name == "Calinski-\nHarabasz\nindex":
            formatted.loc[row_name, "Value"] = f"{value:.0f}"
        else:
            formatted.loc[row_name, "Value"] = f"{value:.3f}"

    return formatted


metric_used = infer_distance_metric(best)
model_performance_table = compute_model_performance_metrics(
    X=X_cluster,
    labels=best["labels"],
    metric=metric_used,
)
model_performance_table_display = format_model_performance_table_for_display(model_performance_table)

print(f"Distance metric used for this table: {metric_used}")

styled_model_performance_table = (
    model_performance_table_display.style
    .set_caption("Model Performance Metrics")
    .set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("text-align", "left"),
                    ("font-style", "italic"),
                    ("font-size", "18px"),
                    ("padding-bottom", "8px"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("font-family", "Arial, sans-serif"),
                    ("font-size", "16px"),
                    ("border-top", "2px solid black"),
                    ("border-bottom", "2px solid black"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "normal"),
                    ("padding", "4px 16px"),
                    ("border-bottom", "2px solid black"),
                ],
            },
            {
                "selector": "tbody th",
                "props": [
                    ("text-align", "left"),
                    ("font-weight", "normal"),
                    ("padding", "6px 16px"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("text-align", "right"),
                    ("padding", "6px 16px"),
                ],
            },
        ]
    )
)

display(styled_model_performance_table)

print(f"Note. All metrics are based on the {metric_used} distance.")

def compute_cluster_means_table(
    X: pd.DataFrame | np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        X_df = X.copy()
    else:
        X = np.asarray(X, dtype=float)
        X_df = pd.DataFrame(X, columns=[f"PC_{i + 1}" for i in range(X.shape[1])])

    labels = np.asarray(labels)
    cluster_order = np.sort(np.unique(labels))

    cluster_means = (
        X_df.assign(Cluster=labels)
        .groupby("Cluster", sort=True)
        .mean()
        .reindex(cluster_order)
    )

    cluster_means.index = [f"Cluster\n{i + 1}" for i in range(len(cluster_means.index))]
    cluster_means.index.name = ""

    return cluster_means


def format_cluster_means_table_for_display(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy().astype(object)
    for col in formatted.columns:
        formatted[col] = formatted[col].map(lambda x: f"{x:.3f}")
    return formatted


cluster_means_table = compute_cluster_means_table(X_cluster, best["labels"])
cluster_means_table_display = format_cluster_means_table_for_display(cluster_means_table)

styled_cluster_means_table = (
    cluster_means_table_display.style
    .set_caption("Cluster Means")
    .set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("caption-side", "top"),
                    ("text-align", "left"),
                    ("font-style", "italic"),
                    ("font-size", "18px"),
                    ("padding-bottom", "8px"),
                ],
            },
            {
                "selector": "table",
                "props": [
                    ("border-collapse", "collapse"),
                    ("font-family", "Arial, sans-serif"),
                    ("font-size", "16px"),
                    ("border-top", "2px solid black"),
                    ("border-bottom", "2px solid black"),
                ],
            },
            {
                "selector": "thead th",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "normal"),
                    ("padding", "6px 16px"),
                    ("border-bottom", "2px solid black"),
                ],
            },
            {
                "selector": "tbody th",
                "props": [
                    ("text-align", "center"),
                    ("font-weight", "normal"),
                    ("padding", "6px 16px"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("text-align", "center"),
                    ("padding", "6px 16px"),
                ],
            },
        ]
    )
)

display(styled_cluster_means_table)
