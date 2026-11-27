"""Replay the published MemSat analyses and regenerate the six paper figures."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, PercentFormatter, ScalarFormatter


HERE = Path(__file__).resolve().parent
FIGURES = HERE.parent / "src" / "figures"
DEPTHS = [1, 2, 4, 8, 16, 32]
BUDGETS = [128, 256, 512]
ABILITY_NAMES = {
    "single-session-assistant": "Assistant",
    "single-session-user": "User",
    "single-session-preference": "Preference",
    "multi-session": "Multi-session",
    "knowledge-update": "Knowledge update",
    "temporal-reasoning": "Temporal",
}
STAMP = "EXPLORATORY / SINGLE READER / NO-GO"


def fit(observations):
    x = pd.DataFrame(index=observations.index)
    x["intercept"] = 1.0
    x["depth"] = np.log2(observations["depth"]) - 2.5
    x["budget"] = np.log2(observations["budget"]) - 8
    x["interaction"] = x["depth"] * x["budget"]
    x["corpus_fraction"] = observations["corpus_fraction"]
    abilities = pd.get_dummies(observations["ability"], dtype=float)
    x = x.join(abilities.drop(columns="knowledge-update"))
    model = sm.GEE(
        observations["em"], x, observations["question"],
        family=sm.families.Binomial(), cov_struct=sm.cov_struct.Independence(),
    )
    return model.fit(ctol=1e-9)


def fit_summary(fit_result):
    ci = fit_result.conf_int()
    model_se = pd.Series(np.sqrt(fit_result.cov_naive.diagonal()), index=ci.index)
    return {
        term: {
            "estimate": float(fit_result.params[term]),
            "robust_se": float(fit_result.bse[term]),
            "model_se": float(model_se[term]),
            "ci95": ci.loc[term].tolist(),
            "p": float(fit_result.pvalues[term]),
        }
        for term in ("depth", "budget", "interaction")
    }


def depth_axis(ax, depths=DEPTHS):
    ax.set_xscale("log", base=2)
    ax.set_xticks(depths)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Retrieval depth")


def format_log_ticks(ax):
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())


def save(fig, filename):
    fig.suptitle(STAMP, fontsize=6.7, fontweight="bold", color="#8b2c38")
    fig.savefig(FIGURES / filename, dpi=300)
    plt.close(fig)


def plot_recall(questions, crossed):
    depths = np.arange(1, 33)
    counts = questions["gold_sessions"].to_numpy()
    gold = crossed[crossed["answerable"]].join(
        questions.set_index("question")["gold_sessions"], on="question"
    )
    matched = gold["retrieved_gold_sessions"]
    counts_for_trial = gold["gold_sessions"]
    observed = gold.assign(
        complete=matched.eq(counts_for_trial), partial=matched / counts_for_trial
    ).groupby("depth")[["complete", "partial"]].mean()
    fig, ax = plt.subplots(figsize=(3.46, 2.02), layout="constrained")
    ax.plot(depths, (counts[:, None] <= depths).mean(axis=0), "--", color="#24537a",
            label="Oracle complete")
    ax.plot(depths, np.minimum(1, depths / counts[:, None]).mean(axis=0), ":",
            color="#895a2b", label="Oracle partial")
    ax.plot(observed.index, observed["complete"], "o-", color="#24537a",
            label="Observed complete")
    ax.plot(observed.index, observed["partial"], "s-", color="#895a2b",
            label="Observed partial")
    depth_axis(ax)
    ax.set_ylabel("Gold-session recall")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0.15, 1.05)
    ax.legend(ncols=2, loc="lower right", fontsize=6.2)
    save(fig, "fig1_recall_ceiling_vs_observed.png")
    return observed


def plot_budget(cells, modeled):
    fig, ax = plt.subplots(figsize=(3.46, 1.85), layout="constrained")
    for _, group in cells.groupby("depth"):
        ax.plot(group["budget"], group["rate"], "o-", color="#7894a9",
                alpha=0.65, linewidth=0.8)
    pooled = modeled.groupby("budget")["em"].mean()
    ax.plot(pooled.index, pooled, "o-", color="#183d64", linewidth=1.5,
            label="Pooled across depths")
    ax.plot([], [], "o-", color="#7894a9", label="Individual depths")
    for budget, value in pooled.items():
        ax.annotate(f"{value:.2%}", (budget, value), xytext=(1, 7),
                    textcoords="offset points", ha="center", color="#183d64")
    ax.set(xscale="log", xticks=BUDGETS, xlabel="Nominal tokens per session",
           ylabel="Exact match", ylim=(0.035, 0.106))
    format_log_ticks(ax)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper left", fontsize=6.2)
    save(fig, "fig2_em_by_budget_b.pdf")


def plot_grid(cells):
    rates = cells["rate"].unstack().loc[DEPTHS, BUDGETS]
    counts = cells["n"].unstack().loc[DEPTHS, BUDGETS]
    fig, ax = plt.subplots(figsize=(3.46, 2.1), layout="constrained")
    im = ax.imshow(rates, cmap="Blues", aspect="auto", vmin=0.025, vmax=0.11)
    for y, depth in enumerate(DEPTHS):
        for x, budget in enumerate(BUDGETS):
            rate = rates.loc[depth, budget]
            ax.text(x, y, f"{rate:.2%}\n(n={counts.loc[depth, budget]})",
                    ha="center", va="center", fontsize=6.8,
                    color="white" if rate > 0.078 else "#172843")
    ax.set(xticks=range(len(BUDGETS)), xticklabels=BUDGETS, xlabel="Nominal tokens per session",
           yticks=range(len(DEPTHS)), yticklabels=DEPTHS, ylabel="Retrieval depth")
    cbar = fig.colorbar(im, ax=ax, label="Exact match")
    cbar.ax.yaxis.set_major_formatter(PercentFormatter(1))
    save(fig, "fig3_kb_em_heatmap.pdf")


def plot_estimates(terms, mde):
    fig, axes = plt.subplots(1, 2, figsize=(3.46, 1.52), layout="constrained",
                             gridspec_kw={"width_ratios": [1.25, 1]})
    for ax in axes:
        ax.axvline(0, color="#666666", linewidth=0.7)
        ax.set_xlabel("Log-odds")
    axes[0].set_yticks([1, 0], ["Depth", "Budget"])
    axes[0].set_ylim(-0.5, 1.5)
    axes[1].set_yticks([0], ["Depth × budget"], fontsize=6.2)
    axes[1].set_xlim(-0.055, 0.065)
    axes[1].axvspan(-mde, mde, color="#c2d5e5", alpha=0.7, label="Planned ±MDE")
    for ax, y, term in ((axes[0], 1, "depth"), (axes[0], 0, "budget"),
                        (axes[1], 0, "interaction")):
        point = terms[term]["estimate"]
        lo, hi = terms[term]["ci95"]
        ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o",
                    markersize=3, capsize=2, color="#183d64")
    axes[1].legend(loc="upper right", fontsize=6.0)
    save(fig, "fig4_interaction_estimate_mde.png")


def plot_abilities(questions, modeled):
    fig, axes = plt.subplots(1, 2, figsize=(3.46, 2.1), layout="constrained")
    colors = ["#1f587f", "#a24e35", "#4e7347", "#795487", "#9c741d", "#484848"]
    markers = ["o", "v", "s", "D", "P", "x"]
    depths = np.arange(1, 7)
    accuracy = modeled.groupby(["ability", "depth"])["em"].mean()
    for (ability, label), color, marker in zip(ABILITY_NAMES.items(), colors, markers):
        observed = accuracy.loc[ability]
        gold = questions.loc[questions["ability"].eq(ability), "gold_sessions"].to_numpy()
        axes[0].plot(observed.index, observed, marker=marker, color=color, label=label)
        axes[1].plot(depths, (gold[:, None] <= depths).mean(axis=0), color=color, marker=marker)
    for ax, label, ticks in zip(axes, ["Exact match", "Oracle complete recall"],
                               [DEPTHS, [1, 2, 4, 6]]):
        depth_axis(ax, ticks)
        ax.set_ylabel(label)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].legend(loc="upper right", fontsize=5.4, labelspacing=0.16)
    save(fig, "fig5_em_by_k_ability.pdf")


def plot_latency(observations, crossed):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    latency = observations["latency_ms"].sort_values().reset_index(drop=True) / 1_000
    quantile = (latency.index + 1) / len(latency)
    axes[0].plot(latency, quantile, color="#183d64")
    axes[0].axhline(0.95, color="#777777", linewidth=0.7, linestyle=":")
    axes[0].set(xscale="log", xticks=[3, 10, 30, 70], xlabel="Response time (s)",
                ylabel="Cumulative share")
    format_log_ticks(axes[0])
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    by_budget = [
        crossed.loc[crossed["budget"].eq(b), "latency_ms"] / 1_000
        for b in BUDGETS
    ]
    axes[1].boxplot(by_budget, tick_labels=BUDGETS, whis=(5, 95), showfliers=False)
    axes[1].set(xlabel="Nominal tokens per session", ylabel="Response time (s)")
    save(fig, "fig6_latency_distribution.png")


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 7.3, "axes.labelsize": 7.3, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 1.1, "lines.markersize": 2.6,
        "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
        "figure.facecolor": "white", "pdf.fonttype": 42,
    })
    questions = pd.read_csv(HERE / "questions.csv")
    observations = pd.read_csv(HERE / "pilot.csv")
    power = json.loads((HERE / "power.json").read_text(encoding="utf-8"))
    gate = json.loads((HERE / "go_no_go.json").read_text(encoding="utf-8"))
    depths = np.arange(1, questions["gold_sessions"].max() + 1)
    coverage = (questions["gold_sessions"].to_numpy()[:, None] <= depths).mean(axis=0)
    saturation_depth = int(depths[coverage >= 0.995][0])
    type_coverage = questions.assign(covered_at_1=questions["gold_sessions"].eq(1))
    type_coverage = type_coverage.groupby("ability")["covered_at_1"].agg(["size", "mean"])
    crossed = observations[observations["arm"].eq("A+G")]
    answerable = crossed[crossed["answerable"]]
    modeled = answerable[~answerable["refused"]]
    assert modeled["em"].isin([0, 1]).all()
    fit_result = fit(modeled)
    assert fit_result.converged
    terms = fit_summary(fit_result)
    no_preference = fit(modeled[modeled["ability"].ne("single-session-preference")])
    assert no_preference.converged
    np.testing.assert_allclose(
        fit_result.params[list(terms)], no_preference.params[list(terms)], atol=1e-8
    )
    np.testing.assert_allclose(
        fit_result.bse[list(terms)], no_preference.bse[list(terms)], atol=1e-8
    )
    sensitivity = {}
    for assumed in (0, 1):
        filled = answerable.assign(em=answerable["em"].mask(answerable["refused"], assumed))
        imputed_fit = fit(filled)
        assert imputed_fit.converged
        sensitivity[assumed] = {
            "em": float(filled["em"].mean()),
            "budget": float(imputed_fit.params["budget"]),
        }
    mde = power["interaction"]["mde_log_odds"]
    cell_em = modeled.groupby(["depth", "budget"])["em"].agg(rate="mean", n="size")
    observed_recall = plot_recall(questions, crossed)
    plot_budget(cell_em.reset_index(), modeled)
    plot_grid(cell_em)
    plot_estimates(terms, mde)
    plot_abilities(questions, modeled)
    plot_latency(observations, crossed)
    print(json.dumps({
        "trials": len(observations),
        "crossed": len(crossed),
        "refusals": int(crossed["refused"].sum()),
        "unanswerable_trials": int((~crossed["answerable"]).sum()),
        "modeled": len(modeled),
        "questions": int(modeled["question"].nunique()),
        "ceiling": dict(zip(map(int, depths), coverage)),
        "ceiling_by_ability_at_1": type_coverage.to_dict("index"),
        "saturation_depth": saturation_depth,
        "median_history_sessions": {
            scale: float(questions[f"sessions_{scale}"].median()) for scale in ("s", "m")
        },
        "full_history_at_64": {
            scale: float(questions[f"sessions_{scale}"].le(64).mean())
            for scale in ("s", "m")
        },
        "em": float(modeled["em"].mean()),
        "em_by_budget": modeled.groupby("budget")["em"].agg(["size", "sum", "mean"]).to_dict("index"),
        "em_by_ability_depth": (
            modeled.groupby(["ability", "depth"])["em"].agg(["size", "sum", "mean"])
            .reset_index().to_dict("records")
        ),
        "missing_first_gold_rank": int(modeled["gold_rank"].isna().sum()),
        "terms": terms,
        "sensitivity": sensitivity,
        "gate": {key: gate[key] for key in ("outcome", "proceed")},
        "planned_design_mde": mde,
        "relative_to_planned_mde": abs(terms["interaction"]["estimate"]) / mde,
        "latency_s": latency_summary(observations["latency_ms"]),
        "latency_s_by_budget": {
            int(b): latency_summary(group["latency_ms"])
            for b, group in crossed.groupby("budget")
        },
        "prefill_512_records": int(observations["prefill_chunk_size"].eq(512).sum()),
        "prefill_unrecorded": int(observations["prefill_chunk_size"].isna().sum()),
        "p95_s_deep_b512": float(
            crossed.query("depth >= 4 and budget == 512")["latency_ms"].quantile(0.95) / 1_000
        ),
        "observed_recall": observed_recall.to_dict(),
    }, indent=2))


def latency_summary(samples_ms):
    q25, median, q75, p95 = samples_ms.quantile([0.25, 0.5, 0.75, 0.95]) / 1_000
    return {"median": float(median), "iqr": float(q75 - q25), "p95": float(p95)}


if __name__ == "__main__":
    main()
