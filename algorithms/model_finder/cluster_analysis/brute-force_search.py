X_cluster = df[["PC1", "PC2"]].copy()

@dataclass
class ClusterConfig:
    cluster_range: tuple = (2, 5)
    algorithms: tuple = ("kmeans", "ward", "complete", "average")
    min_cluster_size: int = 50
    balance_weight: float = 0.25
    top_n: int = 10
    random_state: int = 42
    n_init: int = 20


def normalize(series, higher=True):
    if series.nunique() <= 1:
        return pd.Series(np.ones(len(series)))
    if higher:
        return (series - series.min()) / (series.max() - series.min())
    return (series.max() - series) / (series.max() - series.min())


def brute_force_cluster(X, config):
    results = []

    for k in range(config.cluster_range[0], config.cluster_range[1] + 1):
        for algo in config.algorithms:

            try:
                if algo == "kmeans":
                    model = KMeans(n_clusters=k, random_state=config.random_state, n_init=config.n_init)
                    labels = model.fit_predict(X)
                else:
                    model = AgglomerativeClustering(n_clusters=k, linkage=algo)
                    labels = model.fit_predict(X)
            except:
                continue

            if len(np.unique(labels)) < 2:
                continue

            counts = pd.Series(labels).value_counts()
            min_size = counts.min()
            max_size = counts.max()

            if min_size < config.min_cluster_size:
                continue

            try:
                sil = silhouette_score(X, labels)
                cal = calinski_harabasz_score(X, labels)
                dav = davies_bouldin_score(X, labels)
            except:
                continue

            balance = min_size / max_size

            results.append({
                "algorithm": algo,
                "k": k,
                "silhouette": sil,
                "calinski": cal,
                "davies": dav,
                "balance": balance,
                "labels": labels
            })

    df = pd.DataFrame(results)

    df["sil_n"] = normalize(df["silhouette"], True)
    df["cal_n"] = normalize(df["calinski"], True)
    df["dav_n"] = normalize(df["davies"], False)
    df["bal_n"] = normalize(df["balance"], True)

    df["score"] = (
        0.4 * df["sil_n"] +
        0.25 * df["cal_n"] +
        0.2 * df["dav_n"] +
        config.balance_weight * df["bal_n"]
    )

    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    return df.head(config.top_n), results

config = ClusterConfig(
    cluster_range=(3, 4),
    min_cluster_size=80,
    top_n=10
)

summary, results = brute_force_cluster(X_cluster, config)

best = summary.iloc[0]

print("\nBest model:")
print(best)

df["Cluster"] = best["labels"]

cluster_fit_table = pd.DataFrame(
    [
        {
            "Clusters": int(best["k"]),
            "N": int(len(X_cluster)),
            "Silhouette": float(best["silhouette"]),
            "Calinski-Harabasz": float(best["calinski"]),
            "Davies-Bouldin": float(best["davies"]),
            "Balance": float(best["balance"]),
            "Algorithm": str(best["algorithm"]),
            "Score": float(best["score"]),
        }
    ]
)

display(cluster_fit_table)
