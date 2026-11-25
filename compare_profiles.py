"""Put several profile_policies.py --json outputs side by side in one table.

    python compare_profiles.py distill_results/profile_v1_iter0490.json \
        distill_results/notrade_iter_0199.profile.json distill_results/trade_iter_0199.profile.json \
        --labels v1 no-trade trade --md distill_results/compare.md

Each cell is the mean over the profile's agents (seats). Rates that are ratios of
totals (accept rate, leader share, robber vs chance) are computed from totals over
all agents, not averaged per agent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mean(P, f):
    vals = [f(p) for p in P]
    vals = [v for v in vals if v is not None and v == v]
    return float(np.mean(vals)) if vals else float("nan")


def total(P, f):
    return float(sum(f(p) for p in P))


def ratio(a, b):
    return a / b if b else float("nan")


ROWS = [
    # (section, label, function(data, profiles) -> value, format)
    ("Outcome", "game length (rounds)", lambda d, P: d["mean_rounds"], "{:.1f}"),
    ("Outcome", "games hitting turn limit", lambda d, P: d["truncated"], "{:.0f}"),
    ("Outcome", "final VP", lambda d, P: mean(P, lambda p: p["final_vp"]), "{:.2f}"),
    ("Outcome", "round of win (winner)", lambda d, P: mean(P, lambda p: p["win_round"]), "{:.1f}"),
    ("Opening", "opening pips (2 settlements)", lambda d, P: mean(P, lambda p: p["opening_pips_per_game"]), "{:.1f}"),
    ("Opening", "ore + wheat share of pips",
     lambda d, P: mean(P, lambda p: p["opening_pip_share"]["ORE"] + p["opening_pip_share"]["WHEAT"]), "{:.0%}"),
    ("Opening", "wood + brick share of pips",
     lambda d, P: mean(P, lambda p: p["opening_pip_share"]["WOOD"] + p["opening_pip_share"]["BRICK"]), "{:.0%}"),
    ("Building", "roads / game", lambda d, P: mean(P, lambda p: p["builds_per_game"]["road"]), "{:.2f}"),
    ("Building", "settlements / game", lambda d, P: mean(P, lambda p: p["builds_per_game"]["settlement"]), "{:.2f}"),
    ("Building", "cities / game", lambda d, P: mean(P, lambda p: p["builds_per_game"]["city"]), "{:.2f}"),
    ("Building", "cities / (cities + settlements)", lambda d, P: mean(P, lambda p: p["city_share_of_vp_builds"]), "{:.0%}"),
    ("Tempo", "3rd settlement (round)", lambda d, P: mean(P, lambda p: p["third_settlement_round"]), "{:.1f}"),
    ("Tempo", "  ...reached in", lambda d, P: mean(P, lambda p: p["reached_third_settlement"]), "{:.0%}"),
    ("Tempo", "1st city (round)", lambda d, P: mean(P, lambda p: p["first_city_round"]), "{:.1f}"),
    ("Tempo", "  ...reached in", lambda d, P: mean(P, lambda p: p["reached_first_city"]), "{:.0%}"),
    ("Resources", "bank trades / game", lambda d, P: mean(P, lambda p: p["bank_trades_per_game"]), "{:.1f}"),
    ("Resources", "  share at 4:1", lambda d, P: mean(P, lambda p: p["bank_ratio_share"]["4"]
                                                      if "4" in p["bank_ratio_share"] else p["bank_ratio_share"].get(4)), "{:.0%}"),
    ("Resources", "cards discarded / game", lambda d, P: mean(P, lambda p: p["cards_discarded_per_game"]), "{:.1f}"),
    ("Resources", "cards discarded / round",
     lambda d, P: mean(P, lambda p: p["cards_discarded_per_game"]) / d["mean_rounds"], "{:.2f}"),
    ("Decisions", "decisions with >1 option", lambda d, P: mean(P, lambda p: p["real_choice_share"]), "{:.0%}"),
    ("Decisions", "end turn when VP build legal", lambda d, P: mean(P, lambda p: p["end_turn_with_vp_build"]), "{:.1%}"),
    ("Robber", "steals while one opponent led", lambda d, P: total(P, lambda p: p["robber"].get("clear_leader_steals", 0)), "{:.0f}"),
    ("Robber", "  from leader / expected by chance",
     lambda d, P: ratio(total(P, lambda p: p["robber"].get("steals_from_leader", 0)),
                        total(P, lambda p: p["robber"].get("chance_from_leader", 0))), "{:.2f}"),
    ("Player trades", "offers / agent / game", lambda d, P: mean(P, lambda p: p["trade"].get("offers", 0) / p["games"]), "{:.1f}"),
    ("Player trades", "completed trades / agent / game", lambda d, P: mean(P, lambda p: p["trade"].get("trades", 0) / p["games"]), "{:.1f}"),
    ("Player trades", "accept rate when asked",
     lambda d, P: ratio(total(P, lambda p: p["trade"].get("accepts", 0)),
                        total(P, lambda p: p["trade"].get("accepts", 0) + p["trade"].get("rejects", 0))), "{:.0%}"),
    ("Player trades", "cancels / offer",
     lambda d, P: ratio(total(P, lambda p: p["trade"].get("cancels", 0)), total(P, lambda p: p["trade"].get("offers", 0))), "{:.2f}"),
    ("Player trades", "share of trades with the leader",
     lambda d, P: ratio(total(P, lambda p: p["trade"].get("trades_with_leader", 0)),
                        total(P, lambda p: p["trade"].get("trades", 0))), "{:.0%}"),
]


def fmt(v, f):
    return "-" if v is None or v != v else f.format(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profiles", nargs="+", help="JSON files written by profile_policies.py --json")
    ap.add_argument("--labels", nargs="+", help="column names (default: file names)")
    ap.add_argument("--md", help="also write the table as Markdown to this file")
    args = ap.parse_args()
    labels = args.labels or [Path(p).stem.replace(".profile", "") for p in args.profiles]
    if len(labels) != len(args.profiles):
        raise SystemExit("--labels needs one name per profile")

    data = [json.loads(Path(p).read_text()) for p in args.profiles]
    trading = [d.get("trading", False) for d in data]
    cols = []
    for d in data:
        P = list(d["profiles"].values())
        cols.append([fmt(f(d, P), spec) if (sec != "Player trades" or d.get("trading")) else "-"
                     for sec, _, f, spec in ROWS])

    header = "| Metric | " + " | ".join(labels) + " |"
    lines = [header, "|---|" + "---|" * len(labels)]
    lines.append("| *trading* | " + " | ".join("on" if t else "off" for t in trading) + " |")
    lines.append("| *checkpoint* | " + " | ".join(str(Path(d.get("checkpoint") or "bots").parent.name + "/"
                                                     + Path(d.get("checkpoint") or "-").stem) for d in data) + " |")
    section = None
    for i, (sec, name, _, _) in enumerate(ROWS):
        if sec == "Player trades" and not any(trading):
            continue
        if sec != section:
            lines.append(f"| **{sec}** |" + " |" * len(labels))
            section = sec
        lines.append(f"| {name} | " + " | ".join(c[i] for c in cols) + " |")
    table = "\n".join(lines)
    print(table)
    if args.md:
        Path(args.md).write_text(
            "# Profile comparison\n\nMeans over the 4 agents of each profile (100 self-play games each, "
            "made with profile_policies.py). Ratios are computed from totals over all agents.\n\n"
            + table + "\n")
        print(f"\nwrote {args.md}")


if __name__ == "__main__":
    main()
