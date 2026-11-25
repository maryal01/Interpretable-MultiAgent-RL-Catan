"""RQ2 figures: compare all strategy-exchange conditions.

    python plot_rq2.py                       # reads results/rq2, writes results/rq2/figures/*.png
    python plot_rq2.py --dir some/other/dir --threshold 0.40

Inputs (all written by run_rq2.sh):
  runs/<condition>_s<seed>/log.jsonl, args.json   per-iteration training log
  eval.csv                                        win rate vs random bots per checkpoint and agent
  profiles/<condition>_s<seed>.json               behaviour profile of the final checkpoint (optional)

Figures (each answers one part of RQ2):
  1_learning_curves.png      win rate vs bots over training            -> learn faster?
  2_diversity.png            JS divergence between agents over training -> stay distinct?
  3_tradeoff.png             learning speed (AUC) vs final diversity    -> both at once?
  4_agent_spread.png         final win rate of every agent              -> lifts all, or copies one?
  5_iters_to_threshold.png   iteration each run reached the threshold   -> how much faster?
  6_game_length.png          self-play game length over training        -> learning, every iteration
  7_interleave_cycle.png     what one centralised phase does            -> mechanism (interleave only)
  8_profiles.png             behaviour vs `none` and agent differences  -> what is learned (needs profiles/)

Condition = run directory name without the _s<seed> suffix. Each condition keeps the
same colour and marker in every figure. Bands and error bars are 95% t-intervals
over seeds.
"""
from __future__ import annotations

import argparse
import csv
import warnings
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

# ------------------------------------------------------------------ style (reference palette, light)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, NEUTRAL = "#e1e0d9", "#c3c2b7", "#f0efec"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
# Fixed assignment so a condition never changes colour between figures or runs.
ORDER = ["none", "interleave", "shared", "blend", "distill",
         "interleave_wins", "interleave_c2", "interleave_c10"]
DIVERGING = LinearSegmentedColormap.from_list("div", ["#184f95", "#6da7ec", NEUTRAL, "#ec8f8e", "#a32b2b"])
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 1.0, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "grid.linestyle": "-", "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.labelcolor": INK2,
    "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def style_of(cond, conds):
    i = ORDER.index(cond) if cond in ORDER else len(ORDER) + sorted(c for c in conds if c not in ORDER).index(cond)
    return SLOTS[i % len(SLOTS)], MARKERS[i % len(MARKERS)]


def mean_ci(rows):
    """rows: list of equal-length sequences (one per seed) -> mean, 95% half-width (nan if 1 seed)."""
    a = np.asarray(rows, dtype=float)
    m = np.nanmean(a, axis=0)
    n = np.sum(~np.isnan(a), axis=0)
    sd = np.nanstd(a, axis=0, ddof=1) if a.shape[0] > 1 else np.full_like(m, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        half = np.where(n > 1, stats.t.ppf(0.975, np.maximum(n - 1, 1)) * sd / np.sqrt(n), np.nan)
    return m, half


def smooth(x, k=5):
    x = np.asarray(x, dtype=float)
    if k <= 1 or len(x) < k:
        return x
    ok = ~np.isnan(x)
    num = np.convolve(np.where(ok, x, 0.0), np.ones(k), mode="valid")
    den = np.convolve(ok.astype(float), np.ones(k), mode="valid")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return np.concatenate([np.full(k - 1, np.nan), out])


# ------------------------------------------------------------------ loading
def load(rdir: Path, opponent: str):
    runs = {}
    for d in sorted((rdir / "runs").glob("*_s*")):
        if not (d / "args.json").exists() or not (d / "log.jsonl").exists():
            continue
        cond, seed = d.name.rsplit("_s", 1)
        recs = [json.loads(l) for l in open(d / "log.jsonl")]
        if not recs:
            continue
        runs[d.name] = {"cond": cond, "seed": int(seed), "args": json.loads((d / "args.json").read_text()),
                        "log": recs, "eval": defaultdict(dict), "profile": None}
        p = rdir / "profiles" / f"{d.name}.json"
        if p.exists():
            runs[d.name]["profile"] = json.loads(p.read_text())
    if (rdir / "eval.csv").exists():
        for r in csv.DictReader(open(rdir / "eval.csv")):
            if r["run"] in runs and r["opponent"] == opponent:
                runs[r["run"]]["eval"][int(r["iteration"])][int(r["agent"])] = float(r["win_rate"])
    conds = defaultdict(list)
    for name, r in runs.items():
        conds[r["cond"]].append(r)
    order = sorted(conds, key=lambda c: (ORDER.index(c) if c in ORDER else len(ORDER), c))
    return runs, {c: conds[c] for c in order}


def curve(run):
    """(iterations, mean win rate of the agents) from the evaluation of one run."""
    its = sorted(i for i, a in run["eval"].items() if a)
    return its, [float(np.mean(list(run["eval"][i].values()))) for i in its]


def js_final(run, k=10):
    return float(np.mean([r["js_diversity"] for r in run["log"][-k:]]))


def legend_handles(shown, conds):
    return [plt.Line2D([], [], color=style_of(c, conds)[0], marker=style_of(c, conds)[1], markersize=7,
                       markeredgecolor=SURFACE, markeredgewidth=1.5, label=c) for c in shown]


def finish(fig, ax, path, conds, title, subtitle, legend=True, shown=None):
    """Title + wrapped subtitle at figure level; legend (bottom) only for conditions drawn."""
    import textwrap
    width_chars = int(fig.get_figwidth() * 15)
    sub = "\n".join(textwrap.wrap(subtitle, width_chars))
    n_lines = sub.count("\n") + 1
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.012, 0.985 - 0.075 * 4.8 / fig.get_figheight(), sub, ha="left", va="top", fontsize=9, color=INK2)
    top = 1 - (0.11 + 0.042 * n_lines) * 4.8 / fig.get_figheight()
    shown = [c for c in (shown if shown is not None else conds)]
    if legend and len(shown) >= 2:
        fig.legend(handles=legend_handles(shown, conds), loc="lower center", ncol=min(len(shown), 5),
                   bbox_to_anchor=(0.5, 0.0), fontsize=9)
        fig.tight_layout(rect=(0, 0.07, 1, top))
    else:
        fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


def seeds_note(conds):
    ns = sorted({len(v) for v in conds.values()})
    return f"{'/'.join(map(str, ns))} seed{'s' if ns != [1] else ''} per condition"


def label_line_ends(ax, items):
    """Direct labels at the right end of each line, nudged apart, with leader lines if moved."""
    if not items or len(items) > 4:
        return
    items = sorted(items, key=lambda t: t[1])
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.05
    placed = []
    for x, y, text in items:
        yy = max(y, placed[-1] + gap) if placed else y
        placed.append(yy)
        ax.annotate(text, xy=(x, y), xytext=(8, (yy - y) / (hi - lo) * ax.bbox.height * 0.75),
                    textcoords="offset points", va="center", fontsize=9, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6) if abs(yy - y) > gap * 0.3 else None)


# ------------------------------------------------------------------ figures
def fig_learning_curves(conds, out, threshold):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ends, shown = [], []
    for c, runs in conds.items():
        curves = [curve(r) for r in runs if r["eval"]]
        if not curves:
            continue
        shown.append(c)
        its = sorted(set.intersection(*[set(i) for i, _ in curves]))
        rows = [[dict(zip(i, m))[x] for x in its] for i, m in curves]
        m, h = mean_ci(rows)
        x = np.array(its) + 1                         # before training -> 0
        color, marker = style_of(c, conds)
        ax.plot(x, m, color=color, marker=marker, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5)
        if not np.all(np.isnan(h)):
            ax.fill_between(x, m - h, m + h, color=color, alpha=0.14, linewidth=0)
        ends.append((x[-1], m[-1], c))
    ax.axhline(0.25, color=MUTED, lw=1)
    ax.text(0.005, 0.25, " chance (25%)", transform=ax.get_yaxis_transform(), va="bottom", fontsize=8, color=MUTED)
    ax.axhline(threshold, color=AXIS, lw=1)
    ax.text(0.005, threshold, f" target ({threshold:.0%})", transform=ax.get_yaxis_transform(), va="bottom",
            fontsize=8, color=MUTED)
    ax.set_xlabel("training iteration (0 = before training)")
    ax.set_ylabel("win rate vs 3 random bots")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    label_line_ends(ax, ends)
    finish(fig, ax, out / "1_learning_curves.png", conds, "Learning speed",
           f"Mean win rate of the 4 agents vs random bots; band = 95% CI, {seeds_note(conds)}", shown=shown)


def interleave_spans(conds):
    for c, runs in conds.items():
        a = runs[0]["args"]
        if a.get("exchange") == "interleave":
            C, D, n = a["central_iters"], a["decentral_iters"], a["iters"]
            return c, [(s + D, s + D + C) for s in range(0, n, C + D)]
    return None, []


def fig_per_iteration(conds, out, key, fname, title, subtitle, ylabel, k=5):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ic, spans = interleave_spans(conds)
    n_max = max(len(r["log"]) for runs in conds.values() for r in runs)
    for s, e in spans:
        if s < n_max:
            ax.axvspan(s - 0.5, min(e, n_max) - 0.5, color=NEUTRAL, lw=0, zorder=0)
    ends = []
    n_max = max(len(r["log"]) for runs in conds.values() for r in runs)
    for c, runs in conds.items():
        n = min(len(r["log"]) for r in runs)
        rows = [smooth([np.nan if r["log"][i][key] is None else r["log"][i][key] for i in range(n)], k)
                for r in runs]
        m, h = mean_ci(rows)
        x = np.arange(n)
        color, _ = style_of(c, conds)
        ax.plot(x, m, color=color)
        if not np.all(np.isnan(h)):
            ax.fill_between(x, m - h, m + h, color=color, alpha=0.14, linewidth=0)
        if np.any(~np.isnan(m)):
            j = np.where(~np.isnan(m))[0][-1]
            ends.append((x[j], m[j], c))
    ax.set_xlim(-0.5, n_max - 0.5)
    ax.set_xlabel("training iteration")
    ax.set_ylabel(ylabel)
    label_line_ends(ax, ends)
    if spans:
        subtitle += f"; gray = `{ic}` centralised phases"
    finish(fig, ax, out / fname, conds, title, subtitle + f", {seeds_note(conds)}")


def fig_tradeoff(conds, out):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    shown = []
    for c, runs in conds.items():
        pts = [(js_final(r), float(np.mean(curve(r)[1]))) for r in runs if r["eval"]]
        if not pts:
            continue
        shown.append(c)
        color, marker = style_of(c, conds)
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, color=color, marker=marker, s=40, alpha=0.45, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        (mx,), (hx,) = mean_ci([[x] for x in xs])
        (my,), (hy,) = mean_ci([[y] for y in ys])
        ax.errorbar(mx, my, xerr=None if np.isnan(hx) else hx, yerr=None if np.isnan(hy) else hy, color=color,
                    lw=1.2, capsize=0, zorder=4)
        ax.scatter([mx], [my], color=color, marker=marker, s=130, edgecolors=SURFACE, linewidths=2, zorder=5)
        ax.annotate(c, (mx, my), xytext=(9, 7), textcoords="offset points", fontsize=9, color=INK)
    ax.set_xlabel("final policy diversity (mean pairwise JS divergence; 0 = identical agents)")
    ax.set_ylabel("learning speed (mean win rate over training)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.text(0.99, 0.99, "fast and distinct", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=9)
    finish(fig, ax, out / "3_tradeoff.png", conds, "Speed vs diversity",
           f"Small = one run, large = condition mean with 95% CI; {seeds_note(conds)}", shown=shown)


def fig_agent_spread(conds, out):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    rng = np.random.default_rng(0)
    data = {}
    for c, runs in conds.items():
        vals = []
        for r in runs:
            its, _ = curve(r)
            if its:
                vals += list(r["eval"][its[-1]].values())
        if vals:
            data[c] = vals
    for k, (c, vals) in enumerate(data.items()):
        color, marker = style_of(c, conds)
        x = k + rng.uniform(-0.15, 0.15, len(vals))
        ax.scatter(x, vals, color=color, marker=marker, s=46, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.plot([k - 0.28, k + 0.28], [np.mean(vals)] * 2, color=INK, lw=2, zorder=4)
    ax.axhline(0.25, color=MUTED, lw=1)
    ax.set_xticks(range(len(data)), list(data), color=INK2)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("final win rate vs 3 random bots")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    finish(fig, ax, out / "4_agent_spread.png", conds, "Does exchange lift every agent?",
           f"Each dot = one agent at the last checkpoint; black bar = mean; gray line = chance; {seeds_note(conds)}",
           legend=False)


def fig_iters_to_threshold(conds, out, threshold):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    max_it = 0
    rows = []
    for c, runs in conds.items():
        for r in runs:
            its, m = curve(r)
            if not its:
                continue
            max_it = max(max_it, its[-1] + 1)
            hit = next((i + 1 for i, v in zip(its, m) if v >= threshold), None)
            rows.append((c, hit))
    never = max_it * 1.08 if max_it else 1
    names = [c for c in conds if any(cc == c for cc, _ in rows)]
    for k, c in enumerate(names):
        color, marker = style_of(c, conds)
        hits = [h for cc, h in rows if cc == c]
        if not hits:
            continue
        reached = [h for h in hits if h is not None]
        ys = np.full(len(hits), k) + np.linspace(-0.12, 0.12, len(hits))
        for y, h in zip(ys, hits):
            if h is None:
                ax.scatter([never], [y], facecolors=SURFACE, edgecolors=color, marker=marker, s=60, linewidths=2,
                           zorder=5, clip_on=False)
            else:
                ax.scatter([h], [y], color=color, marker=marker, s=60, edgecolors=SURFACE, linewidths=1.5,
                           zorder=5, clip_on=False)
        if reached:
            med = float(np.median(reached))
            ax.plot([med, med], [k - 0.25, k + 0.25], color=INK, lw=2, zorder=4)
        ax.text(never * 1.03, k, f"{len(reached)}/{len(hits)} reached", va="center", fontsize=8, color=INK2)
    ax.axvline(never, color=AXIS, lw=1)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(-never * 0.04, never * 1.25)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"first evaluated iteration with mean win rate ≥ {threshold:.0%}  (hollow = never)")
    finish(fig, ax, out / "5_iters_to_threshold.png", conds, "How soon each run reaches the target",
           f"Dot = one run; black bar = median of the runs that reached it; evaluated every 20 iterations",
           legend=False)


def fig_interleave_cycle(conds, out):
    ic = next((c for c, runs in conds.items() if runs[0]["args"].get("exchange") == "interleave"), None)
    if ic is None:
        return
    a = conds[ic][0]["args"]
    C, D = a["central_iters"], a["decentral_iters"]
    L = C + D
    series = {"js_diversity": defaultdict(list), "frames_per_game": defaultdict(list)}
    for c, runs in conds.items():
        if runs[0]["args"].get("exchange") != "interleave" or runs[0]["args"]["central_iters"] != C:
            continue
        for r in runs:
            log = r["log"]
            for start in range(0, len(log) - L + 1, L):
                for key in series:
                    base = log[start][key]
                    for p in range(L):
                        v = log[start + p][key]
                        if base is not None and v is not None and v == v and base == base:
                            series[key][p].append(v - base)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    color, marker = style_of(ic, conds)
    titles = {"js_diversity": "change in policy diversity (JS)", "frames_per_game": "change in steps per game"}
    for ax, key in zip(axes, series):
        ps = np.arange(L)
        m = np.array([np.mean(series[key][p]) if series[key][p] else np.nan for p in ps])
        h = np.array([stats.t.ppf(0.975, len(series[key][p]) - 1) * np.std(series[key][p], ddof=1)
                      / np.sqrt(len(series[key][p])) if len(series[key][p]) > 1 else np.nan for p in ps])
        ax.axvspan(D - 0.5, L - 0.5, color=NEUTRAL, lw=0, zorder=0)
        ax.axhline(0, color=AXIS, lw=1)
        ax.plot(ps, m, color=color, marker=marker, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2)
        if not np.all(np.isnan(h)):
            ax.fill_between(ps, m - h, m + h, color=color, alpha=0.14, linewidth=0)
        ax.set_xlim(-0.5, L - 0.5)
        ax.set_xlabel("iteration within cycle")
        ax.set_ylabel(titles[key])
        ax.text(D + C / 2 - 0.5, 1.0, "central", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color=MUTED)
    n_cycles = len(series["js_diversity"][0])
    finish(fig, axes[0], out / "7_interleave_cycle.png", {ic: None}, f"One `{ic}` cycle, averaged",
           f"{D} independent + {C} centralised iterations; change since cycle start, mean of {n_cycles} cycles "
           f"(all seeds) with 95% CI; gray = centralised phase", legend=False)


PROFILE_METRICS = [
    ("game length (rounds)", lambda d, p: d["mean_rounds"]),
    ("opening pips", lambda d, p: p["opening_pips_per_game"]),
    ("3rd settlement round", lambda d, p: p["third_settlement_round"]),
    ("bank trades / game", lambda d, p: p["bank_trades_per_game"]),
    ("discards / game", lambda d, p: p["cards_discarded_per_game"]),
    ("roads / game", lambda d, p: p["builds_per_game"]["road"]),
    ("city share", lambda d, p: p["city_share_of_vp_builds"]),
    ("final VP", lambda d, p: p["final_vp"]),
]


def fig_profiles(conds, out):
    have = {c: [r["profile"] for r in runs if r["profile"]] for c, runs in conds.items()}
    have = {c: v for c, v in have.items() if v}
    if not have:
        return
    names = [m for m, _ in PROFILE_METRICS]
    level = {}      # condition -> metric means (over agents and seeds)
    spread = {}     # condition -> coefficient of variation between agents within a run, averaged over seeds
    for c, profs in have.items():
        per_seed, cv = [], []
        for d in profs:
            agents = [[f(d, p) for _, f in PROFILE_METRICS] for p in d["profiles"].values()]
            a = np.array(agents, dtype=float)
            per_seed.append(np.nanmean(a, axis=0))
            with np.errstate(invalid="ignore", divide="ignore"):
                cv.append(np.nanstd(a, axis=0) / np.abs(np.nanmean(a, axis=0)))
        level[c] = np.nanmean(per_seed, axis=0)
        spread[c] = np.nanmean(cv, axis=0)
    rows = list(have)
    base = level.get("none")
    fig, axes = plt.subplots(1, 2, figsize=(12, 1.2 + 0.55 * len(rows)), gridspec_kw={"width_ratios": [1, 1]})
    if base is not None:
        rel = np.array([(level[c] - base) / np.abs(base) for c in rows])
        lim = max(0.05, float(np.nanmax(np.abs(rel))))
        axes[0].imshow(rel, cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
        txt = [[f"{v:+.0%}" for v in row] for row in rel]
        title0 = "Behaviour relative to `none` (blue = lower, red = higher)"
    else:
        rel = np.array([level[c] for c in rows])
        axes[0].imshow(rel, cmap=SEQUENTIAL, aspect="auto")
        txt = [[f"{v:.2g}" for v in row] for row in rel]
        title0 = "Behaviour (no `none` run to compare against)"
    for i, row in enumerate(txt):
        for j, t in enumerate(row):
            axes[0].text(j, i, t, ha="center", va="center", fontsize=8, color=INK)
    sp = np.array([spread[c] for c in rows])
    axes[1].imshow(sp, cmap=SEQUENTIAL, vmin=0, vmax=max(0.05, float(np.nanmax(sp))), aspect="auto")
    for i, row in enumerate(sp):
        for j, v in enumerate(row):
            axes[1].text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                         color="#ffffff" if v > 0.6 * np.nanmax(sp) else INK)
    for ax, t in zip(axes, (title0, "How much the 4 agents differ (std / mean between agents)")):
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows)), rows)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(t, fontsize=10, pad=8)
    fig.suptitle("What the agents learned (final checkpoints, self-play profile)", x=0.01, ha="left",
                 fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "8_profiles.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {out / '8_profiles.png'}")


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="results/rq2")
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--opponent", default="random")
    args = ap.parse_args()
    rdir = Path(args.dir)
    out = rdir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    runs, conds = load(rdir, args.opponent)
    if not runs:
        raise SystemExit(f"no runs with log.jsonl under {rdir / 'runs'}")
    print(f"{len(runs)} runs: " + ", ".join(f"{c} ({len(v)})" for c, v in conds.items()))
    if any(r["eval"] for r in runs.values()):
        fig_learning_curves(conds, out, args.threshold)
        fig_tradeoff(conds, out)
        fig_agent_spread(conds, out)
        fig_iters_to_threshold(conds, out, args.threshold)
    else:
        print("  no eval.csv rows yet: skipping figures 1, 3, 4, 5")
    fig_per_iteration(conds, out, "js_diversity", "2_diversity.png", "Do the agents stay distinct?",
                      "Mean pairwise JS divergence between the 4 policies on the same states (5-iteration mean)",
                      "policy diversity (JS divergence)")
    fig_per_iteration(conds, out, "frames_per_game", "6_game_length.png", "Self-play game length",
                      "Environment steps per finished self-play game (5-iteration mean); falling = faster play",
                      "steps per game")
    fig_interleave_cycle(conds, out)
    fig_profiles(conds, out)


if __name__ == "__main__":
    main()
