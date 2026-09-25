"""Regenerate every report figure and table from results/*.json and saved runs (invariant 11).

    python -m molfp.figures   -> reports/figures/*.png, reports/tables.md

t-SNE and saliency need a model forward pass, so they load the seed-0 checkpoints; every
number printed in a figure or table comes from results/*.json.
"""

import csv
import io
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from molfp.baseline import ecfp  # noqa: E402
from molfp.data import ROOT, load_dataset, load_graphs  # noqa: E402
from molfp.explain import BLUE, RED, draw_png, saliency  # noqa: E402
from molfp.split import get_split  # noqa: E402
from molfp.train import DEVICE, RUNS, load_model, predict  # noqa: E402

FIG = ROOT / "reports" / "figures"
RESULTS = ROOT / "results"
INK, GRAPHITE, GRID = "#1C2321", "#5E6B67", "#DCE3E0"
GCN_C, RF_C = "#2F6FB0", "#C2703D"  # validated pair (dataviz validator, CVD ΔE 19.4)
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#9DBBDA", "#2F6FB0", "#132F4F"])
DIV = LinearSegmentedColormap.from_list("saliency", [BLUE, (1, 1, 1), RED])

plt.rcParams.update(
    {
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": GRAPHITE,
        "xtick.color": GRAPHITE,
        "ytick.color": GRAPHITE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def load(name):
    return json.loads((RESULTS / name).read_text())


def pm(s, key, fmt=".3f"):
    return f"{s[key]['mean']:{fmt}} ± {s[key]['std']:{fmt}}"


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"[figures] reports/figures/{name}")


def parity(esol):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), sharex=True, sharey=True)
    r0 = esol["gcn"]["per_seed"][0]
    y = np.ravel(r0["test_true"])
    lo, hi = y.min() - 0.5, y.max() + 0.5
    for ax, model, color, label in [
        (axes[0], "gcn", GCN_C, "GCN"),
        (axes[1], "rf", RF_C, "ECFP4 + RF"),
    ]:
        pred = np.ravel(esol[model]["per_seed"][0]["test_pred"])
        ax.plot([lo, hi], [lo, hi], color=GRAPHITE, lw=1, ls="--", zorder=1)
        ax.scatter(y, pred, s=22, color=color, edgecolor="white", linewidth=0.8, zorder=2)
        ax.set_title(
            f"{label}\ntest RMSE {pm(esol[model]['summary'], 'rmse')} (3 seeds)",
            fontsize=10,
            loc="left",
        )
        ax.set_xlabel("measured logS")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
    axes[0].set_ylabel("predicted logS")
    fig.suptitle("ESOL test set, scaffold split, seed 0 shown", x=0.07, ha="left", fontsize=11)
    save(fig, "parity_esol.png")


def read_log(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    return rows[0], np.array(rows[1:], dtype=float)


def training_curves():
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex="row")
    shades = ["#132F4F", "#2F6FB0", "#8DB3DA"]
    for row, ds in enumerate(["esol", "tox21"]):
        for seed, shade in enumerate(shades):
            header, log = read_log(RUNS / ds / "gcn" / f"seed{seed}" / "log.csv")
            for col in (0, 1):
                axes[row, col].plot(
                    log[:, 0], log[:, col + 1], color=shade, lw=1.6, label=f"seed {seed}"
                )
        axes[row, 0].set_ylabel(f"{ds.upper()} train loss")
        axes[row, 1].set_ylabel(header[2].replace("_", " "))
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    axes[0, 1].legend()
    fig.suptitle("GCN training curves (early stopping on the val metric)", x=0.07, ha="left")
    save(fig, "training_curves.png")


def bars(ax, labels, means, stds, color):
    x = np.arange(len(labels))
    ax.bar(
        x, means, yerr=stds, color=color, width=0.6, capsize=3, error_kw={"ecolor": INK, "lw": 1}
    )
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)


def ablations(abl):
    v = abl["variants"]
    groups = [
        ("Depth (GCN layers)", ["depth_1", "depth_2", "depth_3", "depth_4"], ["1", "2", "3", "4"]),
        (
            "Readout",
            ["readout_mean", "readout_max", "readout_meanmax"],
            ["mean", "max", "mean+max"],
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (title, keys, labels) in zip(axes[:2], groups, strict=True):
        bars(
            ax,
            labels,
            [v[k]["gcn"]["rmse"]["mean"] for k in keys],
            [v[k]["gcn"]["rmse"]["std"] for k in keys],
            GCN_C,
        )
        ax.set_title(title, loc="left", fontsize=10)
    ax = axes[2]
    x = np.arange(2)
    for off, model, color, label in [
        (-0.18, "gcn", GCN_C, "GCN"),
        (0.18, "rf", RF_C, "ECFP4 + RF"),
    ]:
        m = [v[k][model]["rmse"]["mean"] for k in ("split_scaffold", "split_random")]
        s = [v[k][model]["rmse"]["std"] for k in ("split_scaffold", "split_random")]
        ax.bar(
            x + off,
            m,
            yerr=s,
            width=0.34,
            color=color,
            capsize=3,
            label=label,
            error_kw={"ecolor": INK, "lw": 1},
        )
    ax.set_xticks(x, ["scaffold", "random"])
    ax.grid(axis="x", visible=False)
    ax.set_title("Split", loc="left", fontsize=10)
    ax.legend(loc="upper right")
    axes[0].set_ylabel("ESOL test RMSE (logS)")
    fig.suptitle("ESOL ablations, mean ± std over 3 seeds", x=0.07, ha="left")
    save(fig, "ablations_esol.png")

    fig, ax = plt.subplots(figsize=(4.5, 3.4))
    keys = ["depth_1", "depth_2", "depth_3", "depth_4"]
    bars(
        ax,
        ["1", "2", "3", "4"],
        [v[k]["gcn"]["rmse"]["mean"] for k in keys],
        [v[k]["gcn"]["rmse"]["std"] for k in keys],
        GCN_C,
    )
    ax.set_xlabel("GCN layers (atoms see this many bonds away)")
    ax.set_ylabel("ESOL test RMSE (logS)")
    ax.set_title("Depth ablation, mean ± std over 3 seeds", loc="left", fontsize=10)
    save(fig, "depth_ablation.png")


def tox21_tasks(tox):
    tasks = tox["tasks"]
    y = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(7, 6))
    for off, model, color, label in [(-0.2, "gcn", GCN_C, "GCN"), (0.2, "rf", RF_C, "ECFP4 + RF")]:
        s = tox[model]["summary"]["roc_auc_per_task"]
        ax.barh(
            y + off,
            s["mean"],
            xerr=s["std"],
            height=0.4,
            color=color,
            capsize=2,
            edgecolor="white",
            linewidth=1,
            label=f"{label}  (mean {pm(tox[model]['summary'], 'roc_auc')})",
            error_kw={"ecolor": INK, "lw": 0.8},
        )
    ax.axvline(0.5, color=GRAPHITE, lw=1, ls="--")
    ax.set_yticks(y, tasks)
    ax.invert_yaxis()
    ax.set_xlim(0.4, 1.0)
    ax.set_xlabel("test ROC-AUC (0.5 = chance)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    ax.set_title(
        "Tox21 per-task ROC-AUC, scaffold split, mean ± std over 3 seeds",
        loc="left",
        fontsize=10,
        pad=26,
    )
    save(fig, "tox21_per_task.png")


def gcn_embeddings(model, graphs):
    model.eval()
    import torch

    with torch.no_grad():
        return np.concatenate(
            [model.embed(b.to(DEVICE)).cpu().numpy() for b in DataLoader(graphs, batch_size=256)]
        )


def tsne_pair():
    smiles, y, _ = load_dataset("esol")
    te = get_split("esol", 0)["test"]
    model, _ = load_model(RUNS / "esol" / "gcn" / "seed0" / "best.pt")
    graphs = load_graphs("esol")
    emb = gcn_embeddings(model, [graphs[i] for i in te])
    bits = ecfp([smiles[i] for i in te]).astype(bool)
    perp = min(30, len(te) // 4)
    z_gcn = TSNE(perplexity=perp, random_state=0, init="pca").fit_transform(emb)
    z_ecfp = TSNE(perplexity=perp, random_state=0, init="random", metric="jaccard").fit_transform(
        bits
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    c = y[te, 0]
    for ax, z, title in [
        (axes[0], z_ecfp, "ECFP4 bits (fixed, Jaccard distance)"),
        (axes[1], z_gcn, "GCN learned fingerprint (256-d)"),
    ]:
        sc = ax.scatter(z[:, 0], z[:, 1], c=c, cmap=SEQ, s=24, edgecolor="white", linewidth=0.6)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.colorbar(sc, ax=axes, shrink=0.85, label="measured logS")
    fig.suptitle(
        "t-SNE of ESOL test molecules (seed-0 split), coloured by solubility", x=0.07, ha="left"
    )
    save(fig, "tsne_pair.png")


def saliency_examples():
    """4 ESOL test molecules spread across the solubility range, drawn with saliency."""
    import matplotlib.image as mpimg

    smiles, y, _ = load_dataset("esol")
    te = get_split("esol", 0)["test"]
    model, ckpt = load_model(RUNS / "esol" / "gcn" / "seed0" / "best.pt")
    graphs = load_graphs("esol")
    pool = sorted((i for i in te if 10 <= graphs[i].num_atoms <= 24), key=lambda i: y[i, 0])
    picks = [pool[int(q * (len(pool) - 1))] for q in (0.1, 0.4, 0.65, 0.9)]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9))
    for ax, i in zip(axes, picks, strict=True):
        s = saliency(model, graphs[i])
        pred = predict(model, [graphs[i]], ckpt["norm"])[0, 0]
        ax.imshow(mpimg.imread(io.BytesIO(draw_png(smiles[i], s, size=(420, 360)))))
        ax.axis("off")
        name = smiles[i] if len(smiles[i]) <= 34 else smiles[i][:31] + "..."
        ax.set_title(f"{name}\nmeasured {y[i, 0]:.2f}   predicted {pred:.2f}", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=DIV, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.05, pad=0.04, aspect=50)
    cb.set_ticks([-1, 0, 1], labels=["lowers predicted logS", "0", "raises predicted logS"])
    fig.suptitle("Gradient × input saliency, ESOL GCN seed 0 (test molecules)", x=0.07, ha="left")
    save(fig, "saliency_examples.png")


def tables(esol, tox, abl):
    """Markdown tables for README/REPORT, straight from results JSON."""
    out = [
        "<!-- generated by python -m molfp.figures; do not edit by hand -->",
        "",
        "**Main results** (test set, scaffold split, mean ± std over seeds 0, 1, 2)",
        "",
        "| Model | ESOL RMSE ↓ | ESOL MAE ↓ | ESOL R² ↑ | Tox21 mean ROC-AUC ↑ |",
        "|---|---|---|---|---|",
    ]
    for model, label in [("gcn", "GCN (learned fingerprint)"), ("rf", "ECFP4 + random forest")]:
        e, t = esol[model]["summary"], tox[model]["summary"]
        out.append(
            f"| {label} | {pm(e, 'rmse')} | {pm(e, 'mae')} | {pm(e, 'r2')} | {pm(t, 'roc_auc')} |"
        )

    out += [
        "",
        "**Tox21 per task** (test ROC-AUC, mean ± std)",
        "",
        "| Task | GCN | ECFP4 + RF |",
        "|---|---|---|",
    ]
    g, r = tox["gcn"]["summary"]["roc_auc_per_task"], tox["rf"]["summary"]["roc_auc_per_task"]
    for k, task in enumerate(tox["tasks"]):
        out.append(
            f"| {task} | {g['mean'][k]:.3f} ± {g['std'][k]:.3f} | "
            f"{r['mean'][k]:.3f} ± {r['std'][k]:.3f} |"
        )

    if abl:
        out += [
            "",
            "**ESOL ablations** (GCN test RMSE, mean ± std; RF shown for the split ablation)",
            "",
            "| Variant | GCN RMSE | RF RMSE |",
            "|---|---|---|",
        ]
        for name, e in abl["variants"].items():
            rf = pm(e["rf"], "rmse") if "rf" in e else "—"
            out.append(f"| {name.replace('_', ' ')} | {pm(e['gcn'], 'rmse')} | {rf} |")
    text = "\n".join(out) + "\n"
    (ROOT / "reports" / "tables.md").write_text(text)
    print("[figures] reports/tables.md")
    return text


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    esol, tox = load("esol.json"), load("tox21.json")
    abl = load("ablations.json") if (RESULTS / "ablations.json").exists() else None
    parity(esol)
    training_curves()
    tox21_tasks(tox)
    if abl:
        ablations(abl)
    else:
        print("[figures] results/ablations.json missing: run `make ablations` for those figures")
    tsne_pair()
    saliency_examples()
    tables(esol, tox, abl)


if __name__ == "__main__":
    main()
