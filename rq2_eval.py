"""RQ2 evaluation: learning speed (win rate vs fixed bots) and diversity per condition.

Step 1 - evaluate checkpoints (resumable; already-evaluated rows are skipped):

    python rq2_eval.py eval results/rq2/runs/* --every 20 --games 40 --workers 4

  For every run directory and every checkpoint whose iteration is a multiple of
  --every (plus iter_before_training and the last one), each agent k plays --games
  games in seat k against 3 `--opponent` bots. All checkpoints use the same game
  seeds (common random numbers), so differences between checkpoints are not luck
  of the deal. Rows go to --csv (default results/rq2/eval.csv).

Step 2 - summarise:

    python rq2_eval.py summary --threshold 0.40

  Per condition (exchange type + its settings), averaged over seeds, with 95% CIs:
    auc            mean win rate over all evaluated checkpoints (higher = learned faster)
    final          win rate at the last checkpoint
    iters_to_thr   first iteration whose 4-agent mean win rate >= --threshold
    js_final       policy diversity at the end (mean pairwise JS divergence, from log.jsonl)
    spread_final   best minus worst agent win rate at the end
  plus the learning curve (mean win rate per iteration) for each condition.
  RQ2 is supported if blend/distill beat `none` on auc/iters_to_thr while keeping
  js_final well above `shared` (which is 0 by construction).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

FIELDS = ["run", "condition", "seed", "iteration", "frames", "agent", "opponent", "games",
          "wins", "win_rate", "mean_rounds", "truncated"]


def condition_label(a: dict) -> str:
    ex = a["exchange"]
    if ex == "blend":
        return f"blend_k{a['every']}_a{a['alpha']}"
    if ex == "distill":
        return f"distill_k{a['every']}_b{a['beta']}_e{a['distill_epochs']}"
    if ex == "interleave":
        return f"interleave_c{a['central_iters']}_d{a['decentral_iters']}_{a['central_weighting']}"
    return ex


def checkpoints(run: Path, every: int):
    its = sorted(int(m.group(1)) for p in run.glob("iter_*.pt") if (m := re.match(r"iter_(\d+)\.pt", p.name)))
    if not its:
        return []
    keep = sorted({i for i in its if i % every == 0} | {its[-1]})
    out = [(-1, run / "iter_before_training.pt")] if (run / "iter_before_training.pt").exists() else []
    return out + [(i, run / f"iter_{i:04d}.pt") for i in keep]


def eval_task(task):
    run, it, path, games, opponent, seed, greedy, threads = task
    import torch
    torch.set_num_threads(threads)
    from catan_env import CatanConfig, CatanCore
    from evaluate import make_policy, play_match
    from runeval import load_policy_net

    a = json.loads((Path(run) / "args.json").read_text())
    config = CatanConfig(num_players=4, map_type="BASE", vps_to_win=8, trading=a["trading"])
    core = CatanCore(config)
    net, _ = load_policy_net(str(path), core)
    torch.manual_seed(seed)
    rows = []
    for k in range(core.n_agents):
        seats = [opponent] * core.n_agents
        seats[k] = make_policy(net, k, core.n_agents, deterministic=greedy)
        res = play_match(seats, games, config, seed=seed)
        rows.append({
            "run": Path(run).name, "condition": condition_label(a), "seed": a["seed"], "iteration": it,
            "frames": (it + 1) * a["frames"], "agent": k, "opponent": opponent, "games": games,
            "wins": int(res.wins[k]), "win_rate": float(res.win_rates[k]),
            "mean_rounds": res.mean_turns / core.n_agents, "truncated": res.truncated,
        })
    return rows


def cmd_eval(args):
    done = set()
    if os.path.exists(args.csv):
        with open(args.csv) as f:
            done = {(r["run"], int(r["iteration"]), r["opponent"]) for r in csv.DictReader(f)}
    tasks = []
    for run in map(Path, args.runs):
        if not (run / "args.json").exists():
            continue
        for it, path in checkpoints(run, args.every):
            if (run.name, it, args.opponent) not in done:
                tasks.append((str(run), it, path, args.games, args.opponent, args.seed, args.greedy, args.threads))
    # Catanatron's move order depends on Python's per-process string hashing, so without a
    # fixed hash seed the same checkpoint + game seeds gives different games in each worker.
    # Workers are new interpreters (spawn) and inherit this environment variable.
    os.environ["PYTHONHASHSEED"] = "0"
    print(f"{len(tasks)} checkpoints to evaluate ({len(done)} already in {args.csv})")
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    new = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f, Pool(args.workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for n, rows in enumerate(pool.imap_unordered(eval_task, tasks), 1):
            w.writerows(rows)
            f.flush()
            r = rows[0]
            print(f"  [{n}/{len(tasks)}] {r['run']} iter {r['iteration']}: "
                  f"win rates {[round(x['win_rate'], 2) for x in rows]}")


# ------------------------------------------------------------------ summary
def ci95(xs):
    xs = np.asarray([x for x in xs if x == x], dtype=float)
    if len(xs) == 0:
        return float("nan"), float("nan"), 0
    if len(xs) == 1:
        return float(xs[0]), float("nan"), 1
    from scipy import stats
    half = stats.t.ppf(0.975, len(xs) - 1) * xs.std(ddof=1) / np.sqrt(len(xs))
    return float(xs.mean()), float(half), len(xs)


def cmd_summary(args):
    rows = list(csv.DictReader(open(args.csv)))
    rows = [r for r in rows if r["opponent"] == args.opponent]
    curve = defaultdict(lambda: defaultdict(list))   # run -> iteration -> agent win rates
    cond_of = {}
    for r in rows:
        curve[r["run"]][int(r["iteration"])].append(float(r["win_rate"]))
        cond_of[r["run"]] = r["condition"]

    per_run = {}
    for run, by_it in curve.items():
        its = sorted(by_it)
        mean = [float(np.mean(by_it[i])) for i in its]
        reached = [i for i, m in zip(its, mean) if m >= args.threshold]
        last = by_it[its[-1]]
        js = float("nan")
        log = Path(args.runs_dir) / run / "log.jsonl"
        if log.exists():
            recs = [json.loads(l) for l in open(log)]
            js = float(np.mean([r["js_diversity"] for r in recs[-10:]])) if recs else js
        per_run[run] = {"its": its, "mean": mean, "auc": float(np.mean(mean)), "final": mean[-1],
                        "iters_to_thr": reached[0] if reached else float("nan"),
                        "reached": bool(reached), "js_final": js, "spread_final": max(last) - min(last)}

    conds = defaultdict(list)
    for run, s in per_run.items():
        conds[cond_of[run]].append(s)
    order = sorted(conds, key=lambda c: (c != "none", c != "shared", c))

    print(f"opponent={args.opponent}, threshold={args.threshold:.2f}, values = mean ± 95% CI over seeds\n")
    cols = [("auc", 3, 17), ("final", 3, 17), ("iters_to_thr", 0, 17), ("reached", None, 9),
            ("js_final", 4, 19), ("spread_final", 3, 17)]
    print(f"{'condition':<26}{'seeds':>6}" + "".join(f"{k:>{w}}" for k, _, w in cols))
    fmt = lambda m, h, d: "-" if m != m else f"{m:.{d}f}" + (f" ± {h:.{d}f}" if h == h else "")
    for c in order:
        S = conds[c]
        line = f"{c:<26}{len(S):>6}"
        for key, d, w in cols:
            if key == "reached":
                cell = f"{sum(s['reached'] for s in S)}/{len(S)}"
            else:
                m, h, _ = ci95([s[key] for s in S])
                cell = fmt(m, h, d)
            line += f"{cell:>{w}}"
        print(line)
    print("(iters_to_thr: -1 means already at the threshold before training)")

    print("\nlearning curve: mean win rate (4 agents, averaged over seeds)")
    all_its = sorted({i for s in per_run.values() for i in s["its"]})
    print(f"{'condition':<26}" + "".join(f"{('init' if i < 0 else i):>7}" for i in all_its))
    for c in order:
        vals = []
        for i in all_its:
            xs = [s["mean"][s["its"].index(i)] for s in conds[c] if i in s["its"]]
            vals.append(f"{np.mean(xs):>7.2f}" if xs else f"{'-':>7}")
        print(f"{c:<26}" + "".join(vals))

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run", "condition", "iteration", "mean_win_rate"])
            for run, s in per_run.items():
                for i, m in zip(s["its"], s["mean"]):
                    w.writerow([run, cond_of[run], i, m])
        print(f"\nper-run curves written to {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("runs", nargs="+", help="run directories written by train_rq2.py")
    e.add_argument("--every", type=int, default=20)
    e.add_argument("--games", type=int, default=40)
    e.add_argument("--opponent", default="random")
    e.add_argument("--seed", type=int, default=10_000, help="first game seed (same for all checkpoints)")
    e.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    e.add_argument("--workers", type=int, default=4)
    e.add_argument("--threads", type=int, default=1, help="torch threads per worker")
    e.add_argument("--csv", default="results/rq2/eval.csv")
    s = sub.add_parser("summary")
    s.add_argument("--csv", default="results/rq2/eval.csv")
    s.add_argument("--runs-dir", default="results/rq2/runs")
    s.add_argument("--opponent", default="random")
    s.add_argument("--threshold", type=float, default=0.40)
    s.add_argument("--out", help="write per-run learning curves to this CSV")
    args = ap.parse_args()
    cmd_eval(args) if args.cmd == "eval" else cmd_summary(args)


if __name__ == "__main__":
    main()
