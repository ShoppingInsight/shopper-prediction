from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "venv" / "Lib" / "site-packages"))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    brier_score_loss,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    make_scorer,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import FunctionTransformer, StandardScaler


OUT_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42

FONT_FAMILY = ["Aptos", "Inter", "Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]
MONO_FONT_FAMILY = ["SF Mono", "Menlo", "Consolas", "DejaVu Sans Mono", "monospace"]

TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLOR_FAMILIES = {
    "blue": {
        "open": TOKENS["panel"],
        "xlight": "#EAF1FE",
        "light": "#CEDFFE",
        "base": "#A3BEFA",
        "mid": "#5477C4",
        "dark": "#2E4780",
    },
    "gold": {
        "open": TOKENS["panel"],
        "xlight": "#FFF4C2",
        "light": "#FFEA8F",
        "base": "#FFE15B",
        "mid": "#B8A037",
        "dark": "#736422",
    },
    "orange": {
        "open": TOKENS["panel"],
        "xlight": "#FFEDDE",
        "light": "#FFBDA1",
        "base": "#F0986E",
        "mid": "#CC6F47",
        "dark": "#804126",
    },
    "olive": {
        "open": TOKENS["panel"],
        "xlight": "#D8ECBD",
        "light": "#BEEB96",
        "base": "#A3D576",
        "mid": "#71B436",
        "dark": "#386411",
    },
    "pink": {
        "open": TOKENS["panel"],
        "xlight": "#FCDAD6",
        "light": "#F5BACC",
        "base": "#F390CA",
        "mid": "#BD569B",
        "dark": "#8A3A6F",
    },
}

NEUTRAL_MARKS = {
    "xlight": "#F4F5F7",
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}


def add_chart_header(
    fig,
    ax,
    title: str,
    subtitle: str,
    *,
    title_width=76,
    subtitle_width=110,
) -> None:
    title = textwrap.fill(str(title).strip(), width=title_width, break_long_words=False)
    subtitle = textwrap.fill(str(subtitle).strip(), width=subtitle_width, break_long_words=False)
    title_lines = title.count("\n") + 1
    subtitle_lines = subtitle.count("\n") + 1
    ax.set_title("")
    fig.subplots_adjust(
        top=max(0.62, 0.86 - 0.045 * (title_lines - 1) - 0.032 * (subtitle_lines - 1))
    )
    left = ax.get_position().x0
    fig.text(
        left,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
        linespacing=1.08,
    )
    fig.text(
        left,
        0.93 - 0.045 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
        linespacing=1.18,
    )
    sns.despine(ax=ax)


def use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.monospace": MONO_FONT_FAMILY,
            "patch.linewidth": 1.0,
        },
    )


def save_fig(fig, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


KNN_LOG_COLS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
]


def log1p_selected_columns(X):
    X = X.copy()
    X[KNN_LOG_COLS] = np.log1p(X[KNN_LOG_COLS])
    return X


def build_knn_pipeline(n_neighbors=5, weights="uniform", p=2):
    return Pipeline(
        [
            ("log1p", FunctionTransformer(log1p_selected_columns, validate=False)),
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            (
                "knn",
                KNeighborsRegressor(
                    n_neighbors=n_neighbors,
                    weights=weights,
                    p=p,
                ),
            ),
        ]
    )


def model_metrics(model_name, y_true, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "model": model_name,
        "ROC_AUC": float(roc_auc_score(y_true, proba)),
        "PR_AUC": float(average_precision_score(y_true, proba)),
        "Brier": float(brier_score_loss(y_true, proba)),
        "Precision": float(precision_score(y_true, pred, zero_division=0)),
        "Recall": float(recall_score(y_true, pred)),
        "F1": float(f1_score(y_true, pred)),
    }


def run() -> None:
    use_chart_theme()

    original = pd.read_csv(ROOT / "data" / "online_shoppers_intention.csv")
    X_train = pd.read_csv(ROOT / "models" / "knn" / "csv" / "X_train_knn.csv")
    X_test = pd.read_csv(ROOT / "models" / "knn" / "csv" / "X_test_knn.csv")
    y_train = pd.read_csv(ROOT / "models" / "knn" / "csv" / "y_train_knn.csv")["Revenue"]
    y_test = pd.read_csv(ROOT / "models" / "knn" / "csv" / "y_test_knn.csv")["Revenue"]

    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)

    class_df = pd.DataFrame(
        {
            "class": ["No purchase", "Purchase"],
            "sessions": [(y_full == 0).sum(), (y_full == 1).sum()],
        }
    )
    class_df["share"] = class_df["sessions"] / class_df["sessions"].sum()

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    colors = [NEUTRAL_MARKS["light"], COLOR_FAMILIES["orange"]["base"]]
    edges = [NEUTRAL_MARKS["dark"], COLOR_FAMILIES["orange"]["dark"]]
    bars = ax.bar(
        class_df["class"],
        class_df["sessions"],
        color=colors,
        edgecolor=edges,
        linewidth=1.0,
    )
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel("")
    ax.set_ylabel("Sessions")
    for bar, share in zip(bars, class_df["share"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + class_df["sessions"].max() * 0.02,
            f"{pct(share)}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=TOKENS["ink"],
        )
    ax.set_ylim(0, class_df["sessions"].max() * 1.18)
    add_chart_header(
        fig,
        ax,
        "Revenue labels are imbalanced",
        "Processed KNN/K-Means input, n=12,205 sessions; purchase class is 15.6% of the data.",
    )
    save_fig(fig, "class_balance")

    raw_model = KNeighborsRegressor(n_neighbors=5, weights="uniform", p=2)
    raw_model.fit(X_train, y_train)
    raw_proba = raw_model.predict(X_test)

    final_model = build_knn_pipeline(n_neighbors=151, weights="distance", p=1)
    final_model.fit(X_train, y_train)
    test_proba = final_model.predict(X_test)

    metrics_df = pd.DataFrame(
        [
            model_metrics("Raw KNN", y_test, raw_proba),
            model_metrics("Pipeline KNN", y_test, test_proba),
        ]
    )

    metrics_long = metrics_df.melt(
        id_vars="model",
        value_vars=["ROC_AUC", "PR_AUC", "Precision", "Recall", "F1"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    model_order = ["Raw KNN", "Pipeline KNN"]
    metric_order = ["ROC_AUC", "PR_AUC", "Precision", "Recall", "F1"]
    palette = {
        "Raw KNN": NEUTRAL_MARKS["base"],
        "Pipeline KNN": COLOR_FAMILIES["blue"]["base"],
    }
    sns.barplot(
        data=metrics_long,
        x="metric",
        y="value",
        hue="model",
        order=metric_order,
        hue_order=model_order,
        palette=palette,
        edgecolor=TOKENS["ink"],
        linewidth=1.0,
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(loc="upper left", frameon=False, ncol=1)
    add_chart_header(
        fig,
        ax,
        "Pipeline preprocessing improves ranking and recall",
        "Held-out test metrics; Brier is excluded because lower is better and belongs in the metric table.",
    )
    save_fig(fig, "knn_metric_comparison")

    test_result = pd.DataFrame(
        {
            "predicted_proba": test_proba,
            "actual_revenue": y_test.reset_index(drop=True),
        }
    )
    test_result["score_decile"] = pd.qcut(
        test_result["predicted_proba"].rank(method="first"), q=10, labels=range(1, 11)
    ).astype(int)
    decile_result = (
        test_result.groupby("score_decile", observed=True)
        .agg(
            count=("actual_revenue", "size"),
            mean_predicted_proba=("predicted_proba", "mean"),
            actual_purchase_rate=("actual_revenue", "mean"),
            actual_buyers=("actual_revenue", "sum"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    line_family = COLOR_FAMILIES["orange"]
    sns.lineplot(
        data=decile_result,
        x="score_decile",
        y="actual_purchase_rate",
        marker="o",
        color=line_family["base"],
        linewidth=1.5,
        ax=ax,
    )
    ax.axhline(y_test.mean(), color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("KNN score decile (10 = highest)")
    ax.set_ylabel("Actual purchase rate")
    ax.text(
        10.05,
        y_test.mean(),
        f" overall {y_test.mean() * 100:.1f}%",
        ha="left",
        va="center",
        fontsize=9,
        color=TOKENS["ink"],
    )
    add_chart_header(
        fig,
        ax,
        "Higher KNN score bands contain more buyers",
        "Held-out test set divided into ten equal score-rank buckets; the score is useful for ordering sessions.",
    )
    save_fig(fig, "knn_decile_purchase_rate")

    top_group = test_result["score_decile"] == 10
    bottom_group = test_result["score_decile"] == 1
    profile_compare = pd.DataFrame(
        {
            "feature": X_test.columns,
            "top_10pct_mean": X_test.loc[top_group].mean().values,
            "bottom_10pct_mean": X_test.loc[bottom_group].mean().values,
        }
    )
    profile_compare["difference"] = (
        profile_compare["top_10pct_mean"]
        - profile_compare["bottom_10pct_mean"]
    )
    profile_compare["abs_difference"] = profile_compare["difference"].abs()
    profile_compare = profile_compare.sort_values("abs_difference", ascending=False)

    perm_scorer = make_scorer(roc_auc_score, response_method="predict")
    importance = permutation_importance(
        final_model,
        X_test,
        y_test,
        scoring=perm_scorer,
        n_repeats=3,
        random_state=RANDOM_STATE,
    )
    importance_result = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    imp_plot = importance_result.head(8).sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    sns.barplot(
        data=imp_plot,
        y="feature",
        x="importance_mean",
        color=COLOR_FAMILIES["gold"]["base"],
        edgecolor=COLOR_FAMILIES["gold"]["dark"],
        linewidth=1.0,
        ax=ax,
    )
    ax.axvline(0, color=TOKENS["ink"], linewidth=1.0)
    ax.set_xlabel("ROC-AUC decrease after shuffling")
    ax.set_ylabel("")
    add_chart_header(
        fig,
        ax,
        "KNN relies on visitor type and session-depth signals",
        "Permutation importance on the held-out test set, three repeats; larger decrease means stronger reliance.",
    )
    save_fig(fig, "knn_permutation_importance")

    LOG_COLS = KNN_LOG_COLS
    X_log = X_full.copy()
    X_log[LOG_COLS] = np.log1p(X_log[LOG_COLS])
    scaler = StandardScaler()
    Z = scaler.fit_transform(X_log)

    sweep = []
    for k in range(2, 11):
        km = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        labels = km.fit_predict(Z)
        sil = silhouette_score(Z, labels, sample_size=4000, random_state=RANDOM_STATE)
        sweep.append({"K": k, "inertia": km.inertia_, "silhouette": sil})
    sweep_df = pd.DataFrame(sweep)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.9))
    axes[0].plot(
        sweep_df["K"],
        sweep_df["inertia"],
        marker="o",
        color=COLOR_FAMILIES["blue"]["base"],
        markeredgecolor=COLOR_FAMILIES["blue"]["dark"],
    )
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("Inertia")
    axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    axes[1].plot(
        sweep_df["K"],
        sweep_df["silhouette"],
        marker="o",
        color=COLOR_FAMILIES["pink"]["base"],
        markeredgecolor=COLOR_FAMILIES["pink"]["dark"],
    )
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("Silhouette")
    for ax_i in axes:
        ax_i.axvline(3, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
        sns.despine(ax=ax_i)
    add_chart_header(
        fig,
        axes[0],
        "K=3 balances separation and interpretability",
        "Elbow is strongest around K=3; K=7 is slightly higher on silhouette but produces a less explainable segmentation.",
    )
    save_fig(fig, "kmeans_k_selection")

    best_k = 3
    kmeans = KMeans(
        n_clusters=best_k,
        random_state=RANDOM_STATE,
        n_init=10,
    )
    cluster_labels = kmeans.fit_predict(Z)
    profile = X_full.copy()
    profile["cluster"] = cluster_labels
    profile["Revenue"] = y_full.values
    profile_table = profile.groupby("cluster").agg(
        size=("Revenue", "size"),
        purchase_rate=("Revenue", "mean"),
        ProductRelated=("ProductRelated", "mean"),
        ProductRelated_Duration=("ProductRelated_Duration", "mean"),
        Administrative=("Administrative", "mean"),
        BounceRates=("BounceRates", "mean"),
        ExitRates=("ExitRates", "mean"),
        is_new_visitor=("is_new_visitor", "mean"),
        Weekend=("Weekend", "mean"),
    )
    profile_table["size_pct"] = profile_table["size"] / profile_table["size"].sum()
    ordered_clusters = profile_table.sort_values("purchase_rate").index.tolist()
    label_map = {
        ordered_clusters[0]: "Bouncers",
        ordered_clusters[1]: "Browsers",
        ordered_clusters[2]: "Engaged",
    }
    profile_table["segment"] = [label_map[c] for c in profile_table.index]

    cluster_plot = profile_table.loc[ordered_clusters].reset_index()
    fig, ax = plt.subplots(figsize=(8.2, 5.1))
    cluster_palette = {
        "Bouncers": COLOR_FAMILIES["orange"]["light"],
        "Browsers": COLOR_FAMILIES["blue"]["base"],
        "Engaged": COLOR_FAMILIES["olive"]["base"],
    }
    cluster_edges = {
        "Bouncers": COLOR_FAMILIES["orange"]["dark"],
        "Browsers": COLOR_FAMILIES["blue"]["dark"],
        "Engaged": COLOR_FAMILIES["olive"]["dark"],
    }
    bars = ax.bar(
        cluster_plot["segment"],
        cluster_plot["purchase_rate"],
        color=[cluster_palette[s] for s in cluster_plot["segment"]],
        edgecolor=[cluster_edges[s] for s in cluster_plot["segment"]],
        linewidth=1.0,
    )
    ax.axhline(y_full.mean(), color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("")
    ax.set_ylabel("Actual purchase rate")
    ax.text(
        2.05,
        y_full.mean(),
        f" overall {y_full.mean() * 100:.1f}%",
        ha="left",
        va="center",
        fontsize=9,
        color=TOKENS["ink"],
    )
    for bar, _, row in zip(
        bars,
        cluster_plot["segment"],
        cluster_plot.itertuples(index=False),
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{row.size_pct * 100:.1f}% of sessions",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TOKENS["ink"],
        )
    ax.set_ylim(0, max(cluster_plot["purchase_rate"]) * 1.32)
    add_chart_header(
        fig,
        ax,
        "Behavior clusters rise from bounce to engagement",
        "K-Means used behavior variables only; Revenue is added afterward for validation and interpretation.",
    )
    save_fig(fig, "kmeans_cluster_purchase_rate")

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Z)
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(coords), size=min(5200, len(coords)), replace=False)
    pca_df = pd.DataFrame(coords[sample_idx], columns=["PC1", "PC2"])
    pca_df["segment"] = [label_map[c] for c in cluster_labels[sample_idx]]
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="segment",
        hue_order=["Bouncers", "Browsers", "Engaged"],
        palette=cluster_palette,
        edgecolor="none",
        alpha=0.42,
        s=16,
        ax=ax,
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=3,
        borderaxespad=0,
    )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
    add_chart_header(
        fig,
        ax,
        "PCA view shows a session-depth axis",
        f"2D projection explains {pca.explained_variance_ratio_.sum() * 100:.1f}% of variance; clustering itself used all 11 variables.",
    )
    save_fig(fig, "kmeans_pca_projection")

    sil_full = silhouette_score(Z, cluster_labels)
    db = davies_bouldin_score(Z, cluster_labels)
    ch = calinski_harabasz_score(Z, cluster_labels)
    sil_by_cluster = (
        pd.DataFrame(
            {
                "cluster": cluster_labels,
                "silhouette": silhouette_samples(Z, cluster_labels),
            }
        )
        .groupby("cluster")
        .agg(mean=("silhouette", "mean"), size=("silhouette", "size"))
    )
    ari = adjusted_rand_score(y_full, cluster_labels)
    nmi = normalized_mutual_info_score(y_full, cluster_labels)

    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X_full,
        y_full,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_full,
    )
    Xtr_log = Xtr_raw.copy()
    Xte_log = Xte_raw.copy()
    Xtr_log[LOG_COLS] = np.log1p(Xtr_log[LOG_COLS])
    Xte_log[LOG_COLS] = np.log1p(Xte_log[LOG_COLS])
    sc_gen = StandardScaler().fit(Xtr_log)
    Z_tr = sc_gen.transform(Xtr_log)
    Z_te = sc_gen.transform(Xte_log)
    km_gen = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(Z_tr)
    lab_tr = km_gen.labels_
    lab_te = km_gen.predict(Z_te)
    sil_tr = silhouette_score(Z_tr, lab_tr)
    sil_te = silhouette_score(Z_te, lab_te)

    def rate_size(labels, yv):
        grouped = (
            pd.DataFrame(
                {
                    "cluster": labels,
                    "Revenue": yv.values,
                }
            )
            .groupby("cluster")["Revenue"]
        )
        return grouped.mean(), grouped.size()

    rt_tr, sz_tr = rate_size(lab_tr, ytr)
    rt_te, sz_te = rate_size(lab_te, yte)
    gen_order = rt_tr.sort_values().index
    gen_df = pd.DataFrame(
        {
            "segment": ["Bouncers", "Browsers", "Engaged"],
            "train_purchase_rate": rt_tr[gen_order].values,
            "test_purchase_rate": rt_te[gen_order].values,
            "train_share": (sz_tr[gen_order] / sz_tr.sum()).values,
            "test_share": (sz_te[gen_order] / sz_te.sum()).values,
        }
    )

    gen_plot = gen_df.melt(
        id_vars="segment",
        value_vars=["train_purchase_rate", "test_purchase_rate"],
        var_name="split",
        value_name="purchase_rate",
    )
    gen_plot["split"] = gen_plot["split"].map(
        {"train_purchase_rate": "Train", "test_purchase_rate": "Held-out test"}
    )
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    sns.barplot(
        data=gen_plot,
        x="segment",
        y="purchase_rate",
        hue="split",
        order=["Bouncers", "Browsers", "Engaged"],
        hue_order=["Train", "Held-out test"],
        palette={
            "Train": COLOR_FAMILIES["blue"]["base"],
            "Held-out test": COLOR_FAMILIES["gold"]["base"],
        },
        edgecolor=TOKENS["ink"],
        linewidth=1.0,
        ax=ax,
    )
    ax.axhline(y_full.mean(), color=TOKENS["ink"], linestyle=":", linewidth=1.0, label="Overall")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.set_xlabel("")
    ax.set_ylabel("Actual purchase rate")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], loc="upper left", frameon=False, ncol=1)
    add_chart_header(
        fig,
        ax,
        "Cluster meaning generalizes to held-out sessions",
        f"Train/test silhouette is nearly identical: {sil_tr:.3f} vs {sil_te:.3f}.",
    )
    save_fig(fig, "kmeans_generalization")

    seed_values = [0, 1, 7, 21, 42, 100, 2024, 99, 13, 77]
    seed_labels = [
        KMeans(n_clusters=best_k, random_state=s, n_init=10).fit_predict(Z)
        for s in seed_values
    ]
    seed_pairs = []
    for i in range(len(seed_labels)):
        for j in range(i + 1, len(seed_labels)):
            seed_pairs.append(adjusted_rand_score(seed_labels[i], seed_labels[j]))

    rng = np.random.default_rng(RANDOM_STATE)
    boot_ari = []
    for _ in range(20):
        idx = rng.choice(len(Z), size=int(0.8 * len(Z)), replace=False)
        boot_labels = KMeans(
            n_clusters=best_k,
            random_state=RANDOM_STATE,
            n_init=10,
        ).fit_predict(Z[idx])
        boot_ari.append(adjusted_rand_score(cluster_labels[idx], boot_labels))

    summary = {
        "data": {
            "original_rows": int(len(original)),
            "processed_rows": int(len(X_full)),
            "features": int(X_full.shape[1]),
            "purchase_rate": float(y_full.mean()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_purchase_rate": float(y_train.mean()),
            "test_purchase_rate": float(y_test.mean()),
            "columns": list(X_full.columns),
        },
        "knn": {
            "selected": {
                "K": 151,
                "weights": "distance",
                "p": 1,
                "distance": "Manhattan",
            },
            "metrics": metrics_df.to_dict(orient="records"),
            "deciles": decile_result.to_dict(orient="records"),
            "top_bottom_profile": profile_compare.to_dict(orient="records"),
            "permutation_importance": importance_result.to_dict(orient="records"),
            "prediction_summary": {
                "min": float(test_proba.min()),
                "max": float(test_proba.max()),
                "mean": float(test_proba.mean()),
                "actual_rate": float(y_test.mean()),
            },
        },
        "kmeans": {
            "selected_K": 3,
            "sweep": sweep_df.to_dict(orient="records"),
            "profile": profile_table.reset_index().to_dict(orient="records"),
            "label_map": {str(k): v for k, v in label_map.items()},
            "pca_explained_variance": [float(x) for x in pca.explained_variance_ratio_],
            "internal_metrics": {
                "silhouette": float(sil_full),
                "davies_bouldin": float(db),
                "calinski_harabasz": float(ch),
                "inertia": float(kmeans.inertia_),
                "silhouette_by_cluster": (
                    sil_by_cluster
                    .reset_index()
                    .to_dict(orient="records")
                ),
            },
            "external_metrics": {
                "ari_vs_revenue": float(ari),
                "nmi_vs_revenue": float(nmi),
            },
            "stability": {
                "seed_ari_mean": float(np.mean(seed_pairs)),
                "seed_ari_min": float(np.min(seed_pairs)),
                "bootstrap_ari_mean": float(np.mean(boot_ari)),
                "bootstrap_ari_min": float(np.min(boot_ari)),
            },
            "generalization": {
                "train_silhouette": float(sil_tr),
                "test_silhouette": float(sil_te),
                "segments": gen_df.to_dict(orient="records"),
            },
        },
    }

    with open(OUT_DIR / "report_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary["data"], ensure_ascii=False, indent=2))
    print(metrics_df.round(4).to_string(index=False))
    print(profile_table.round(4).to_string())


if __name__ == "__main__":
    run()
