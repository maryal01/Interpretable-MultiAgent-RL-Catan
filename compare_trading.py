"""RQ2: do the exchange strategies help as much with player trading on as with it off?

    python compare_trading.py                         # none vs interleave, 200-iteration folders
    python compare_trading.py --off results/rq2_long_notrade --on results/rq2_long_trading \
        --strategies none shared blend distill interleave --out results/rq2_long_compare

Reads two run_rq2.sh output folders (same settings except --trading) and compares each
strategy with the `none` baseline in each setting. No training or evaluation is run;
everything comes from runs/*/log.jsonl, eval.csv and profiles/*.json. `none` must be
among --strategies (it is the baseline).

Encoding in every figure: colour + marker = strategy (as in plot_rq2.py); solid line /
filled marker = trading off, dashed / hollow = trading on.

Writes to --out:
  figures/1_learning_curves.png   win rate vs random bots, one panel per setting
  figures/2_gain_vs_none.png      strategy minus none, per checkpoint, one panel per strategy
  figures/3_tradeoff.png          learning speed vs final diversity, every strategy x setting
  figures/4_diversity.png         JS divergence between agents over training, per setting
  figures/5_game_length.png       self-play steps per game over training, per setting
  figures/6_interleave_cycle.png  what one centralised phase does (only if interleave is included)
  figures/7_profiles.png          behaviour change vs none, every strategy x setting
  summary.md                      tables behind the figures
"""
from __future__ import annotations

import argparse
import textwrap
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from plot_rq2 import (AXIS, DIVERGING, INK, INK2, MUTED, NEUTRAL, SURFACE, curve, js_final, load, mean_ci,
                      smooth, style_of)

STRATS = ["none", "interleave"]
SETTINGS = [("off", "-", True), ("on", "--", False)]      # name, line style, filled marker


# ------------------------------------------------------------------ helpers
def handles(strats=STRATS, settings=True):
    h = [plt.Line2D([], [], color=style_of(s, {})[0], marker=style_of(s, {})[1], markersize=8, ls="",
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=s) for s in strats]
    if settings:
        h += [plt.Line2D([], [], color=INK2, ls=ls, lw=2, label=f"trading {name}") for name, ls, _ in SETTINGS]
    return h


def finish(fig, path, title, subtitle, legend_handles=None):
    width_chars = int(fig.get_figwidth() * 15)
    sub = "\n".join(textwrap.wrap(subtitle, width_chars))
    n_lines = sub.count("\n") + 1
    fig.text(0.012, 0.985, title, ha="left", va="top", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.012, 0.985 - 0.075 * 4.8 / fig.get_figheight(), sub, ha="left", va="top", fontsize=9, color=INK2)
    top = 1 - (0.11 + 0.042 * n_lines) * 4.8 / fig.get_figheight()
    if legend_handles:
        fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
                   bbox_to_anchor=(0.5, 0.0), fontsize=9)
        fig.tight_layout(rect=(0, 0.07, 1, top))
    else:
        fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  wrote {path}")


def end_labels(ax, items, gap_frac=0.055):
    """items: (x, y, text, colour). Labels to the right of line ends, nudged apart vertically."""
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * gap_frac
    placed = []
    for x, y, text in sorted(items, key=lambda t: t[1]):
        yy = max(y, placed[-1] + gap) if placed else y
        placed.append(yy)
        ax.annotate(text, xy=(x, y), xytext=(x, yy), textcoords="data", va="center", fontsize=8.5, color=INK2,
                    xycoords="data", annotation_clip=False)


def run_curve_rows(runs):
    """Aligned (x, rows) of 4-agent mean win rate, x = iteration + 1 (0 = before training)."""
    curves = [curve(r) for r in runs if r["eval"]]
    its = sorted(set.intersection(*[set(i) for i, _ in curves]))
    rows = [[dict(zip(i, m))[x] for x in its] for i, m in curves]
    return np.array(its) + 1, rows


def welch(a, b):
    """mean(a) - mean(b) with a 95% Welch t half-width (a, b independent samples)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1)) if se > 0 else 1
    return a.mean() - b.mean(), stats.t.ppf(0.975, df) * se


def run_metrics(r, threshold):
    its, m = curve(r)
    hit = next((i + 1 for i, v in zip(its, m) if v >= threshold), None)
    final = r["eval"][its[-1]]
    return {"auc": float(np.mean(m)), "final": m[-1], "iters_to_thr": hit, "js_final": js_final(r),
            "spread_final": max(final.values()) - min(final.values())}


# ------------------------------------------------------------------ figures
def fig_learning_curves(data, out, threshold):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    ends_by_ax = []
    for ax, (setting, ls, _) in zip(axes, SETTINGS):
        ends = []
        for s in STRATS:
            x, rows = run_curve_rows(data[setting][s])
            m, h = mean_ci(rows)
            color, marker = style_of(s, {})
            ax.plot(x, m, color=color, ls=ls, marker=marker, markersize=6, markeredgecolor=SURFACE,
                    markeredgewidth=1.5)
            ax.fill_between(x, m - h, m + h, color=color, alpha=0.14, linewidth=0)
            ends.append((x[-1], m[-1], f"  {s} {m[-1]:.0%}"))
        ax.axhline(0.25, color=MUTED, lw=1)
        ax.axhline(threshold, color=AXIS, lw=1)
        ax.text(0.55, 0.25, "chance (25%)", transform=ax.get_yaxis_transform(), va="top", fontsize=8, color=MUTED)
        ax.text(0.03, threshold, f"target ({threshold:.0%})", transform=ax.get_yaxis_transform(), va="top",
                fontsize=8, color=MUTED)
        ax.set_title(f"Trading {setting}", fontsize=11)
        ax.set_xlabel("training iteration (0 = before training)")
        ax.set_xlim(-0.02 * x[-1], x[-1] * 1.22)
        ends_by_ax.append((ax, ends))
    axes[0].set_ylabel("win rate vs 3 random bots")
    for ax, ends in ends_by_ax:
        end_labels(ax, ends)
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    finish(fig, out / "1_learning_curves.png", "Learning speed with and without trading",
           "Mean win rate of the 4 agents vs random bots; band = 95% CI over 3 seeds; same evaluation game seeds "
           "in both panels", handles())


def fig_gain(data, out):
    others = [t for t in STRATS if t != "none"]
    fig, axes = plt.subplots(1, len(others), figsize=(max(8, 4.2 * len(others)), 4.6), sharey=True, squeeze=False)
    for ax, strat in zip(axes[0], others):
        color, marker = style_of(strat, {})
        ends = []
        for setting, ls, filled in SETTINGS:
            x, ri = run_curve_rows(data[setting][strat])
            _, rn = run_curve_rows(data[setting]["none"])
            g = np.array([welch(np.array(ri)[:, j], np.array(rn)[:, j]) for j in range(len(x))])
            ax.plot(x, g[:, 0], color=color, ls=ls, marker=marker, markersize=5, lw=2,
                    markerfacecolor=color if filled else SURFACE, markeredgecolor=color if not filled else SURFACE,
                    markeredgewidth=1.3)
            ax.fill_between(x, g[:, 0] - g[:, 1], g[:, 0] + g[:, 1], color=color, alpha=0.10 if filled else 0.06,
                            linewidth=0)
            ends.append((x[-1], g[-1, 0], f"  {setting} {g[-1, 0]:+.0%}"))
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xlim(-0.02 * x[-1], x[-1] * 1.25)
        ax.set_title(f"{strat} minus none", fontsize=11)
        ax.set_xlabel("training iteration")
        end_labels(ax, ends, gap_frac=0.07)
    axes[0][0].set_ylabel("difference in win rate vs random bots")
    axes[0][0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    finish(fig, out / "2_gain_vs_none.png", "How much each strategy adds over independent learning",
           "Difference in mean win rate vs random bots, strategy minus none in the same setting; band = 95% Welch CI "
           "(3 + 3 seeds); above 0 = strategy ahead",
           [plt.Line2D([], [], color=INK2, ls=ls, lw=2, label=f"trading {n}") for n, ls, _ in SETTINGS])


def fig_tradeoff(data, out):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    means = {}
    for setting, _, filled in SETTINGS:
        for s in STRATS:
            pts = [(js_final(r), float(np.mean(curve(r)[1]))) for r in data[setting][s]]
            color, marker = style_of(s, {})
            xs, ys = zip(*pts)
            fc = color if filled else SURFACE
            ax.scatter(xs, ys, marker=marker, s=40, facecolors=fc, edgecolors=color, alpha=0.45, linewidths=1.5,
                       zorder=3)
            (mx,), (hx,) = mean_ci([[v] for v in xs])
            (my,), (hy,) = mean_ci([[v] for v in ys])
            ax.errorbar(mx, my, xerr=hx, yerr=hy, color=color, lw=1.2, capsize=0, zorder=4)
            ax.scatter([mx], [my], marker=marker, s=130, facecolors=fc, edgecolors=color if not filled else SURFACE,
                       linewidths=2, zorder=5)
            means[(setting, s)] = (mx, my)
    for setting, _, _ in SETTINGS:
        x0, y0 = means[(setting, "none")]
        for strat in STRATS:
            x1, y1 = means[(setting, strat)]
            if strat != "none":
                ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                            arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9, shrinkA=9, shrinkB=9))
            if len(STRATS) > 2:
                ax.annotate(strat, (x1, y1), xytext=(8, 6), textcoords="offset points", fontsize=8, color=INK2)
        ax.annotate(f"trading {setting}", (x0, y0), xytext=(0, -24), textcoords="offset points",
                    ha="center", va="top", fontsize=10, fontweight="bold", color=INK)
    ax.set_xlabel("final policy diversity (mean pairwise JS divergence; 0 = identical agents)")
    ax.set_ylabel("learning speed (mean win rate over training)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.text(0.99, 0.99, "fast and distinct", transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=9)
    hs = handles(settings=False) + [
        plt.Line2D([], [], ls="", marker="o", markersize=8, markerfacecolor=INK2, markeredgecolor=INK2,
                   label="trading off"),
        plt.Line2D([], [], ls="", marker="o", markersize=8, markerfacecolor=SURFACE, markeredgecolor=INK2,
                   markeredgewidth=1.5, label="trading on")]
    finish(fig, out / "3_tradeoff.png", "Speed vs diversity, with and without trading",
           "Small = one run, large = mean of 3 seeds with 95% CI; arrows = effect of switching from none to each strategy",
           hs)


def fig_per_iteration(data, out, key, fname, title, subtitle, ylabel, share_y, k=5):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=share_y)
    for ax, (setting, ls, _) in zip(axes, SETTINGS):
        n_it = data[setting]["none"][0]["args"]["iters"]
        if "interleave" in STRATS:
            a = data[setting]["interleave"][0]["args"]
            C, D = a["central_iters"], a["decentral_iters"]
            for s0 in range(0, n_it, C + D):
                ax.axvspan(s0 + D - 0.5, min(s0 + D + C, n_it) - 0.5, color=NEUTRAL, lw=0, zorder=0)
        for s in STRATS:
            runs = data[setting][s]
            n = min(len(r["log"]) for r in runs)
            rows = [smooth([np.nan if r["log"][i][key] is None else r["log"][i][key] for i in range(n)], k)
                    for r in runs]
            m, h = mean_ci(rows)
            color, _ = style_of(s, {})
            ax.plot(np.arange(n), m, color=color, ls=ls)
            ax.fill_between(np.arange(n), m - h, m + h, color=color, alpha=0.14 if len(STRATS) <= 2 else 0.07,
                            linewidth=0)
        ax.set_title(f"Trading {setting}", fontsize=11)
        ax.set_xlabel("training iteration")
        ax.set_xlim(-0.5, n_it - 0.5)
        if not share_y or ax is axes[0]:
            ax.set_ylabel(ylabel)
    finish(fig, out / fname, title, subtitle + ("; gray = interleave centralised phases" if "interleave" in STRATS else "")
           + "; band = 95% CI over 3 seeds",
           handles())


def cycle_changes(runs, key, relative):
    a = runs[0]["args"]
    C, D = a["central_iters"], a["decentral_iters"]
    L = C + D
    vals = [[] for _ in range(L)]
    for r in runs:
        log = r["log"]
        for start in range(0, len(log) - L + 1, L):
            base = log[start][key]
            for p in range(L):
                v = log[start + p][key]
                if base is None or v is None or v != v or base != base:
                    continue
                vals[p].append((v - base) / base if relative else v - base)
    m = np.array([np.mean(v) for v in vals])
    h = np.array([stats.t.ppf(0.975, len(v) - 1) * np.std(v, ddof=1) / np.sqrt(len(v)) for v in vals])
    return C, D, m, h, len(vals[0])


def fig_cycle(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    color, marker = style_of("interleave", {})
    specs = [("js_diversity", False, "change in policy diversity (JS)"),
             ("frames_per_game", True, "change in steps per game (% of cycle start)")]
    for ax, (key, rel, ylabel) in zip(axes, specs):
        for setting, ls, filled in SETTINGS:
            C, D, m, h, n = cycle_changes(data[setting]["interleave"], key, rel)
            ps = np.arange(C + D)
            ax.plot(ps, m, color=color, ls=ls, marker=marker, markersize=5, markeredgewidth=1.2,
                    markerfacecolor=color if filled else SURFACE, markeredgecolor=SURFACE if filled else color)
            ax.fill_between(ps, m - h, m + h, color=color, alpha=0.10 if filled else 0.06, linewidth=0)
        ax.axvspan(D - 0.5, C + D - 0.5, color=NEUTRAL, lw=0, zorder=0)
        ax.axhline(0, color=AXIS, lw=1)
        ax.text(D + C / 2 - 0.5, 1.0, "central", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8, color=MUTED)
        ax.set_xlim(-0.5, C + D - 0.5)
        ax.set_xlabel("iteration within cycle")
        ax.set_ylabel(ylabel)
        if rel:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    finish(fig, out / "6_interleave_cycle.png", "One interleave cycle, with and without trading",
           f"{D} independent + {C} centralised iterations; change since cycle start, mean of {n} cycles per "
           "setting with 95% CI; gray = centralised phase",
           [plt.Line2D([], [], color=color, ls=ls, marker=marker, markersize=6, lw=2,
                       markerfacecolor=color if f else SURFACE, markeredgecolor=color, label=f"trading {nm}")
            for nm, ls, f in SETTINGS])


PROFILE = [
    ("game length (rounds)", lambda d, p: d["mean_rounds"], "{:.1f}"),
    ("opening pips", lambda d, p: p["opening_pips_per_game"], "{:.1f}"),
    ("3rd settlement round", lambda d, p: p["third_settlement_round"], "{:.1f}"),
    ("bank trades / game", lambda d, p: p["bank_trades_per_game"], "{:.1f}"),
    ("discards / game", lambda d, p: p["cards_discarded_per_game"], "{:.1f}"),
    ("roads / game", lambda d, p: p["builds_per_game"]["road"], "{:.2f}"),
    ("city share", lambda d, p: p["city_share_of_vp_builds"], "{:.0%}"),
    ("final VP", lambda d, p: p["final_vp"], "{:.2f}"),
]
TRADE = [
    ("offers / agent / game", lambda d, p: p["trade"]["offers"] / p["games"], "{:.1f}"),
    ("completed trades / agent / game", lambda d, p: p["trade"]["trades"] / p["games"], "{:.1f}"),
    ("accept rate", lambda d, p: p["trade"]["accepts"] / max(1, p["trade"]["accepts"] + p["trade"]["rejects"]),
     "{:.0%}"),
    ("trades with the leader", lambda d, p: p["trade"]["trades_with_leader"] / max(1, p["trade"]["trades"]),
     "{:.0%}"),
]


def profile_level(runs, metrics):
    """Mean over agents, then over seeds."""
    per_seed = []
    for r in runs:
        d = r["profile"]
        if d is None:
            continue
        per_seed.append(np.nanmean([[f(d, p) for _, f, _ in metrics] for p in d["profiles"].values()], axis=0))
    return np.nanmean(per_seed, axis=0) if per_seed else None


def fig_profiles(data, out):
    names = [m for m, _, _ in PROFILE] + [m for m, _, _ in TRADE]
    rel, labels = [], []
    for strat in [t for t in STRATS if t != "none"]:
        for setting, _, _ in SETTINGS:
            ms = PROFILE + (TRADE if setting == "on" else [])
            li, ln = profile_level(data[setting][strat], ms), profile_level(data[setting]["none"], ms)
            row = list((li - ln) / np.abs(ln))
            rel.append(row + [np.nan] * (len(names) - len(row)))
            labels.append(f"{strat}, trading {setting}")
    rel = np.array(rel)
    lim = max(0.05, float(np.nanmax(np.abs(rel))))
    fig, ax = plt.subplots(figsize=(12, 2.4 + 0.5 * len(rel)))
    ax.imshow(np.ma.masked_invalid(rel), cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto")
    for i, row in enumerate(rel):
        for j, v in enumerate(row):
            ax.text(j, i, "n/a" if np.isnan(v) else f"{v:+.0%}", ha="center", va="center", fontsize=8,
                    color=MUTED if np.isnan(v) else INK)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels)
    for k in range(2, len(labels), 2):
        ax.axhline(k - 0.5, color=SURFACE, lw=4)
    ax.axvline(len(PROFILE) - 0.5, color=SURFACE, lw=4)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    finish(fig, out / "7_profiles.png", "What each strategy changes in behaviour",
           "Final checkpoints, 100 self-play games each; strategy relative to none in the same setting "
           "(blue = lower, red = higher); mean over 4 agents and 3 seeds; trade columns only exist with trading on")


# ------------------------------------------------------------------ summary table
def summary_md(data, out, threshold):
    L = ["# RQ2 strategies with and without player trading", "",
         f"Generated by `compare_trading.py` from `{data['dirs'][0]}` (trading off) and `{data['dirs'][1]}` "
         f"(trading on). {len(data['on']['none'])} seeds per group, {data['on']['none'][0]['args']['iters']} iterations "
         "x 6,000 frames, `simple` reward; win rate vs 3 "
         "`random` bots, 40 games per agent per checkpoint. Values: mean ± 95% CI over seeds.", "",
         "## Learning speed and diversity", "",
         "| trading | strategy | auc | final | iters to 40% | js_final | spread_final |", "|---|---|---|---|---|---|---|"]
    per = {}
    for setting, _, _ in SETTINGS:
        for s in STRATS:
            ms = [run_metrics(r, threshold) for r in data[setting][s]]
            per[(setting, s)] = ms
            cells = []
            for k, f in [("auc", "{:.3f}"), ("final", "{:.1%}"), ("iters_to_thr", "{:.0f}"), ("js_final", "{:.4f}"),
                         ("spread_final", "{:.2f}")]:
                v = [m[k] for m in ms if m[k] is not None]
                (mu,), (hw,) = mean_ci([[x] for x in v])
                cells.append(f"{f.format(mu)} ± {f.format(hw)}")
            L.append(f"| {setting} | {s} | " + " | ".join(cells) + " |")
    L += ["", "## Gain over none (strategy minus none in the same setting, Welch 95% CI)", "",
          "| strategy | trading | Δ auc | Δ final | Δ js_final | js kept |", "|---|---|---|---|---|---|"]
    for strat in [t for t in STRATS if t != "none"]:
        for setting, _, _ in SETTINGS:
            cells = []
            for k, f in [("auc", "{:+.3f}"), ("final", "{:+.1%}"), ("js_final", "{:+.4f}")]:
                d, h = welch([m[k] for m in per[(setting, strat)]], [m[k] for m in per[(setting, "none")]])
                cells.append(f"{f.format(d)} ± {f.format(h).lstrip('+')}")
            kept = np.mean([m["js_final"] for m in per[(setting, strat)]]) / \
                np.mean([m["js_final"] for m in per[(setting, "none")]])
            L.append(f"| {strat} | {setting} | " + " | ".join(cells) + f" | {kept:.0%} |")
    cols = [(st, t) for st, _, _ in SETTINGS for t in STRATS]
    L += ["", "## Behaviour of the final checkpoints (mean over agents and seeds)", "",
          "| metric | " + " | ".join(f"{t}, {st}" for st, t in cols) + " |", "|---|" + "---|" * len(cols)]
    lv = {(st, s): profile_level(data[st][s], PROFILE + (TRADE if st == "on" else []))
          for st, _, _ in SETTINGS for s in STRATS}
    for j, (name, _, f) in enumerate(PROFILE + TRADE):
        cells = []
        for st, _, _ in SETTINGS:
            for s in STRATS:
                v = lv[(st, s)]
                cells.append(f.format(v[j]) if v is not None and j < len(v) else "-")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\n  wrote {out / 'summary.md'}")


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--off", default="results/rq2", help="run_rq2.sh folder with trading off")
    ap.add_argument("--on", default="results/rq2_trading", help="run_rq2.sh folder with trading on")
    ap.add_argument("--out", default="results/rq2_trading_compare")
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--strategies", nargs="+", default=["none", "interleave"],
                    help="conditions to compare (run folder names without _s<seed>); must include none")
    args = ap.parse_args()
    if "none" not in args.strategies:
        raise SystemExit("--strategies must include none (the baseline)")
    STRATS[:] = ["none"] + [t for t in args.strategies if t != "none"]
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    data = {}
    for setting, d in (("off", args.off), ("on", args.on)):
        _, conds = load(Path(d), "random")
        missing = [s for s in STRATS if s not in conds]
        if missing:
            raise SystemExit(f"{d}: no runs for {missing}")
        data[setting] = {s: conds[s] for s in STRATS}
        print(f"trading {setting}: " + ", ".join(f"{s} ({len(conds[s])} seeds)" for s in STRATS))
    data["dirs"] = (args.off, args.on)
    figs = out / "figures"
    fig_learning_curves(data, figs, args.threshold)
    fig_gain(data, figs)
    fig_tradeoff(data, figs)
    fig_per_iteration(data, figs, "js_diversity", "4_diversity.png", "Do the agents stay distinct?",
                      "Mean pairwise JS divergence between the 4 policies on the same states (5-iteration mean)",
                      "policy diversity (JS divergence)", share_y=True)
    fig_per_iteration(data, figs, "frames_per_game", "5_game_length.png", "Self-play game length",
                      "Environment steps per finished self-play game (5-iteration mean); trade actions count as "
                      "steps, so compare within a panel, not across", "steps per game", share_y=False)
    if "interleave" in STRATS:
        fig_cycle(data, figs)
    fig_profiles(data, figs)
    summary_md(data, out, args.threshold)


if __name__ == "__main__":
    main()
