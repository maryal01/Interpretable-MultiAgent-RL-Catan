"""Classify and summarize logged robber decisions for RQ1.

Run from the repository:
    python summarize_leader_targeting.py logs/rq1_run1/robber_decisions.csv

Prints summary rates to stdout; does not modify input files.
Use --format json for a machine-readable summary or --format decisions for
one JSON object per classified decision (the original step-1 output).
Primary: unique highest-visible-VP player across the whole game.
Secondary: unique highest-visible-VP opponent of the acting player.
Both measures require a choice between stealing from the leader and another
opponent. Ties, unreachable leaders, and choices without another opponent are
excluded. The primary measure also excludes decisions made by the leader.
For eligible decisions the baseline is uniform over ALL legal robber actions,
including actions without a victim; it is not uniform over opponent players.
Excluded decisions have null outcomes and baselines, rather than counting as
failures to target. This measures theft victims, not tile-blocking damage.

Rates weight eligible decisions equally. Expected targets are the sum of their
individual legal-action probabilities; the baseline rate is their mean, not a
ratio of pooled legal-action counts. The difference is in percentage points.
This is a descriptive comparison conditional on the observed game states, not
a simulation of random-agent games, a significance test, or proof of cooperation.
95% confidence intervals use 10,000 whole-game bootstrap resamples by default.
Each resample recomputes the decision-weighted difference, keeping outcomes and
baselines paired. The games.csv inventory is used when available, including
games with zero robber decisions. Otherwise only logged games are resampled.
Intervals describe evaluation-game variation for a fixed policy, not variation
across training runs; estimates from very few games should be treated as pilot
results. Use --bootstrap-samples and --bootstrap-seed to control resampling.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from math import fsum
from pathlib import Path
import random


def _classify(scores, actor, victim, actions):
    highest = max(scores.values())
    leaders = [color for color, vp in scores.items() if vp == highest]
    leader = leaders[0] if len(leaders) == 1 else None
    leader_count = sum(action["victim_color"] == leader for action in actions) if leader else 0
    alternative = any(action["victim_color"] not in (None, leader) for action in actions)

    if leader is None:
        reason = "tied_leaders"
    elif leader == actor:
        reason = "actor_is_leader"
    elif leader_count == 0:
        reason = "leader_unreachable"
    elif not alternative:
        reason = "no_alternative_opponent"
    else:
        reason = None

    eligible = reason is None
    return {
        "leader_color": leader,
        "eligible": eligible,
        "exclusion_reason": reason,
        "targeted_leader": victim == leader if eligible else None,
        "random_legal_action_leader_probability": leader_count / len(actions) if eligible else None,
    }


def classify_decision(row):
    """Classify a CSV row using pre-move VP and actual legal tile/victim pairs.

    Recomputes both definitions from raw evidence; the logger's older leader
    columns describe opponents only and must not be treated as primary labels.
    """
    scores = json.loads(row["visible_vps"])
    actor = row["actor_color"]
    victim = row["victim_color"] or None
    tile = json.loads(row["selected_tile"])
    actions = json.loads(row["legal_robber_actions"])

    if not isinstance(scores, dict) or len(scores) < 2 or actor not in scores:
        raise ValueError("visible_vps must contain the actor and at least one opponent")
    if any(type(vp) is not int or vp < 0 for vp in scores.values()):
        raise ValueError("Visible victory points must be nonnegative integers")
    if not isinstance(actions, list) or not actions:
        raise ValueError("At least one legal robber action is required")
    for action in actions:
        if not isinstance(action, dict) or not {"tile", "victim_color"} <= action.keys():
            raise ValueError("Each legal action needs a tile and victim_color")
        candidate = action["victim_color"]
        if candidate is not None and (candidate not in scores or candidate == actor):
            raise ValueError("A legal victim must be an opponent or null")
    if not any(action["tile"] == tile and action["victim_color"] == victim for action in actions):
        raise ValueError("Selected robber action is absent from the legal actions")

    opponents = {color: vp for color, vp in scores.items() if color != actor}
    return {
        "primary": _classify(scores, actor, victim, actions),
        "secondary": _classify(opponents, actor, victim, actions),
    }


def classify_file(path):
    """Load and validate all rows before returning results for display."""
    results = []
    identities = set()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for line, row in enumerate(csv.DictReader(stream), start=2):
            try:
                identity = (int(row["game_seed"]), int(row["decision_index"]))
                if identity in identities:
                    raise ValueError("Duplicate game_seed/decision_index")
                identities.add(identity)
                results.append({
                    "game_index": int(row["game_index"]),
                    "game_seed": identity[0],
                    "decision_index": identity[1],
                    "actor_color": row["actor_color"],
                    **classify_decision(row),
                })
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}, CSV line {line}: {error}") from error
    if not results:
        raise ValueError(f"{path}: no robber decisions found")
    return results


def _summarize_group(decisions, measure, actor):
    eligible = [decision for decision in decisions if decision[measure]["eligible"]]
    count = len(eligible)
    targeted = sum(decision[measure]["targeted_leader"] for decision in eligible)
    expected = fsum(decision[measure]["random_legal_action_leader_probability"] for decision in eligible)
    observed_rate = targeted / count if count else None
    baseline_rate = expected / count if count else None
    return {
        "measure": measure,
        "actor_color": actor,  # null denotes all actors pooled
        "total_decisions": len(decisions),
        "games_with_logged_decisions": len({decision["game_seed"] for decision in decisions}),
        "games_with_eligible_decisions": len({decision["game_seed"] for decision in eligible}),
        "eligible_decisions": count,
        "excluded_decisions": len(decisions) - count,
        "exclusion_reasons": dict(sorted(Counter(
            decision[measure]["exclusion_reason"] for decision in decisions
            if not decision[measure]["eligible"]
        ).items())),
        "observed_leader_targets": targeted,
        "expected_leader_targets": expected,
        "observed_targeting_rate": observed_rate,
        "random_legal_action_baseline_rate": baseline_rate,
        "excess_percentage_points": 100 * (observed_rate - baseline_rate) if count else None,
    }


def load_game_seeds(path, decisions):
    """Validate the game inventory, including games with zero robber decisions."""
    counts = Counter(decision["game_seed"] for decision in decisions)
    seeds = set()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for line, row in enumerate(csv.DictReader(stream), start=2):
            try:
                seed = int(row["game_seed"])
                if seed in seeds:
                    raise ValueError("Duplicate game_seed")
                seeds.add(seed)
                if int(row["robber_decision_count"]) != counts.get(seed, 0):
                    raise ValueError("Robber count disagrees with decision log")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{path}, CSV line {line}: {error}") from error
    if not seeds or not counts.keys() <= seeds:
        raise ValueError("Game inventory is empty or omits games in the decision log")
    return sorted(seeds)


def _percentile(values, probability):
    """Linear interpolation between sorted order statistics."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def add_bootstrap_intervals(summary, decisions, game_seeds, samples, seed):
    """Paired whole-game percentile bootstrap of the decision-weighted excess.

    Draw G games with replacement from the G original games per replicate.
    Each occurrence contributes all eligible decisions, including all actors
    in pooled estimates. Recompute the ratio using the sampled denominator.
    Share draws across all groups; do not sample individual moves or agents.
    Zero-eligible games remain in the sampling pool. Undefined replicates are
    counted and omitted from percentiles; fewer than two informative games
    cannot support a between-game interval and receive null bounds.
    """
    if type(samples) is not int or samples < 2:
        raise ValueError("bootstrap_samples must be an integer of at least 2")
    if type(seed) is not int or seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    seeds = sorted(set(game_seeds))
    if not seeds or not {d["game_seed"] for d in decisions} <= set(seeds):
        raise ValueError("Bootstrap game inventory must contain every logged game")
    groups = summary["summaries"]
    totals = []
    for group in groups:
        buckets = {game: [] for game in seeds}
        for decision in decisions:
            if group["actor_color"] is not None and decision["actor_color"] != group["actor_color"]:
                continue
            value = decision[group["measure"]]
            if value["eligible"]:
                buckets[decision["game_seed"]].append(
                    int(value["targeted_leader"]) - value["random_legal_action_leader_probability"]
                )
        totals.append([(len(buckets[game]), fsum(buckets[game])) for game in seeds])

    rng = random.Random(seed)
    distributions = [[] for _ in groups]
    for _ in range(samples):
        indices = rng.choices(range(len(seeds)), k=len(seeds))
        for totals_for_group, distribution in zip(totals, distributions):
            count = sum(totals_for_group[index][0] for index in indices)
            if count:
                difference = fsum(totals_for_group[index][1] for index in indices)
                distribution.append(100 * difference / count)

    for group, distribution in zip(groups, distributions):
        if group["games_with_eligible_decisions"] < 2:
            status = "fewer_than_two_eligible_games"
        elif len(distribution) < 2:
            status = "fewer_than_two_valid_resamples"
        else:
            status = "ok"
        group["excess_ci_95_pp"] = {
            "lower": _percentile(distribution, 0.025) if status == "ok" else None,
            "upper": _percentile(distribution, 0.975) if status == "ok" else None,
            "status": status,
            "valid_resamples": len(distribution),
            "undefined_resamples": samples - len(distribution),
        }
    summary["bootstrap"] = {
        "method": "whole-game percentile bootstrap; linear-interpolated 2.5th and 97.5th percentiles",
        "confidence_level": 0.95,
        "samples": samples,
        "seed": seed,
        "game_count": len(seeds),
        "game_seeds": seeds,
        "undefined_resamples": "Replicates with no eligible decisions are omitted, not assigned zero excess.",
        "scope": "Pointwise intervals for this fixed policy and evaluation setup, assuming independent games; not uncertainty across training runs or simultaneous agent comparisons.",
    }


def summarize_decisions(decisions, *, game_seeds=None, bootstrap_samples=10000, bootstrap_seed=0):
    """Summarize classify_file output; zero eligible choices yield null rates.

    All-agent rates pool decisions rather than averaging the per-agent rates.
    Logged/eligible game counts cover the supplied robber log. An optional
    game_seeds inventory also retains games with no robber decisions for the
    bootstrap; the CLI reads this inventory from games.csv when present.
    """
    actors = sorted({decision["actor_color"] for decision in decisions})
    groups = [(None, decisions)] + [
        (actor, [decision for decision in decisions if decision["actor_color"] == actor])
        for actor in actors
    ]
    summary = {
        "schema_version": 2,
        "definitions": {
            "primary": "Unique highest visible VP across all players; actor must not be the leader.",
            "secondary": "Unique highest visible VP among the actor's opponents.",
            "eligibility": "Leader and at least one other opponent are legal theft victims.",
            "baseline": "Uniform over all legal robber tile/victim actions, including no-victim actions, in each observed state.",
            "aggregation": "Decision-weighted rates over eligible decisions; baseline is the mean of their individual probabilities.",
            "missing_rates": "Null means no eligible decisions; it does not mean a zero targeting rate.",
            "scope": "Conditional legal-action comparison; intervals resample games, not training runs. Logged-decision game counts exclude games without robber decisions.",
        },
        "summaries": [
            _summarize_group(group, measure, actor)
            for measure in ("primary", "secondary") for actor, group in groups
        ],
    }
    inventory = game_seeds if game_seeds is not None else {d["game_seed"] for d in decisions}
    add_bootstrap_intervals(summary, decisions, inventory, bootstrap_samples, bootstrap_seed)
    summary["bootstrap"]["inventory_source"] = "provided game inventory" if game_seeds is not None else "games represented in the robber log only"
    return summary


def format_summary(summary):
    """Human-readable rates; full-precision values remain available as JSON."""
    lines = [
        "Leader targeting (eligible robber decisions only)",
        "Primary: overall game leader. Secondary: highest-scoring opponent.",
        "",
        f"{'Measure':<10} {'Actor':<8} {'Eligible':>8} {'Targets':>8} {'Observed':>10} {'Baseline':>10} {'Excess (pp)':>12} {'95% CI (pp)':>20}",
    ]
    for row in summary["summaries"]:
        observed = row["observed_targeting_rate"]
        baseline = row["random_legal_action_baseline_rate"]
        excess = row["excess_percentage_points"]
        observed_text = f"{observed:.2%}" if observed is not None else "N/A"
        baseline_text = f"{baseline:.2%}" if baseline is not None else "N/A"
        excess_text = f"{excess:+.2f}" if excess is not None else "N/A"
        interval = row["excess_ci_95_pp"]
        interval_text = f"[{interval['lower']:+.2f}, {interval['upper']:+.2f}]" if interval["status"] == "ok" else "N/A"
        lines.append(
            f"{row['measure']:<10} {row['actor_color'] or 'ALL':<8} "
            f"{row['eligible_decisions']:>8} {row['observed_leader_targets']:>8} "
            f"{observed_text:>10} {baseline_text:>10} {excess_text:>12} {interval_text:>20}"
        )
    for row in summary["summaries"]:
        if row["actor_color"] is None:
            reasons = ", ".join(f"{reason}={count}" for reason, count in row["exclusion_reasons"].items()) or "none"
            lines.append(f"\n{row['measure']}: {row['excluded_decisions']}/{row['total_decisions']} excluded ({reasons}).")
    lines.extend([
        "\nBaseline: average chance of targeting the leader under uniform legal actions in the observed states.",
        "Excess = observed rate minus baseline, in percentage points (pp). N/A = no eligible decisions.",
        f"95% CI: whole-game percentile bootstrap, {summary['bootstrap']['samples']} resamples, seed {summary['bootstrap']['seed']}, {summary['bootstrap']['game_count']} games.",
        f"Game inventory: {summary['bootstrap']['inventory_source']}.",
        "Intervals use game-to-game variation for this fixed policy; few games can give unstable intervals.",
        "Intervals are pointwise, not adjusted for comparisons across agents; they do not establish deliberate cooperation.",
    ])
    for row in summary["summaries"]:
        interval = row["excess_ci_95_pp"]
        if interval["status"] != "ok" or interval["undefined_resamples"]:
            lines.append(f"{row['measure']} {row['actor_color'] or 'ALL'} CI: {interval['status']}; "
                         f"{interval['undefined_resamples']} resamples omitted because they had no eligible decisions.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("decisions", type=Path, help="Existing robber_decisions.csv")
    parser.add_argument("--format", choices=("summary", "json", "decisions"), default="summary",
                        help="Readable summary (default), JSON summary, or original per-decision JSON lines.")
    parser.add_argument("--bootstrap-samples", type=int, default=10000, help="Whole-game resamples (default: 10000).")
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="Fixed resampling seed for reproducible intervals.")
    parser.add_argument("--games-file", type=Path, help="Game inventory; defaults to games.csv beside the decision log if present.")
    args = parser.parse_args()
    try:
        results = classify_file(args.decisions)
        if args.format != "decisions":
            games_path = args.games_file or args.decisions.with_name("games.csv")
            use_inventory = args.games_file is not None or games_path.exists()
            seeds = load_game_seeds(games_path, results) if use_inventory else None
            summary = summarize_decisions(results, game_seeds=seeds,
                                          bootstrap_samples=args.bootstrap_samples, bootstrap_seed=args.bootstrap_seed)
            if use_inventory:
                summary["bootstrap"]["inventory_source"] = str(games_path.resolve())
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    if args.format == "decisions":
        for result in results:
            print(json.dumps(result))
    else:
        if args.format == "json":
            print(json.dumps({"source": str(args.decisions.resolve()), **summary}, indent=2, allow_nan=False))
        else:
            print(format_summary(summary))


if __name__ == "__main__":
    main()
