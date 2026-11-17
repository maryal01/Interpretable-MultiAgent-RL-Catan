"""Log executed robber decisions for RQ1 using four trained policies in self-play.

Example:
    python assess_leader_targeting.py --checkpoint checkpoints/iter_0399.pt \
        --games 10 --seed 10000 --output-dir logs/rq1_run1

Leader means the unique highest-visible-VP opponent immediately before a move.
All robber decisions are retained, including ties and moves without a victim.
The random baseline is uniform over ALL legal robber actions (tile/victim pairs),
not over players. This script collects evidence; it does not test significance
or establish cooperation. Run from the repository with its environment active.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import torch

from catanatron.models.enums import ActionType
from catanatron.observer import GameObserver
from catanatron.state_functions import get_visible_victory_points

from catan_env import CatanCore
from evaluate import PolicyPlayer, make_policy
from runeval import CONFIG, load_policy_net


DECISION_FIELDS = [
    "game_index", "game_seed", "decision_index", "completed_turns",
    "agent_index", "actor_color", "actor_seating_position", "turn_order",
    "victim_color", "victim_relative_offset", "selected_tile",
    "visible_vps", "overall_leaders", "opponent_leaders",
    "unique_opponent_leader", "highest_opponent_vp", "opponent_vp_gap",
    "opponent_leader_tied", "leader_reachable", "leader_targeting_forced",
    "alternative_opponent_reachable", "informative_victim_choice",
    "targeted_unique_leader", "legal_robber_action_count",
    "legal_leader_action_count", "random_legal_action_leader_probability",
    "legal_robber_actions",
]
GAME_FIELDS = [
    "game_index", "game_seed", "turn_order", "winner_color", "terminated",
    "truncated", "completed_turns", "decision_count", "robber_decision_count",
    "final_visible_vps",
]


def json_cell(value):
    return json.dumps(value, separators=(",", ":"))


def robber_decision_row(game, action, *, game_index, game_seed, agent_colors):
    """Describe a selected engine action from pre-action state, or return None.

Engine Actions contain actual colors; do not decode victims from the RL action
table, whose color labels stand for positions relative to the actor.
"""
    if action.action_type != ActionType.MOVE_ROBBER:
        return None
    legal = [a for a in game.playable_actions if a.action_type == ActionType.MOVE_ROBBER]
    if action not in legal:
        raise ValueError("Cannot log a robber action that is not currently legal")

    order = list(game.state.colors)
    visible = {color: get_visible_victory_points(game.state, color) for color in order}
    opponents = {color: vp for color, vp in visible.items() if color != action.color}
    highest = max(opponents.values())
    leaders = [color for color, vp in opponents.items() if vp == highest]
    leader = leaders[0] if len(leaders) == 1 else None
    ordered_vps = sorted(opponents.values(), reverse=True)
    gap = ordered_vps[0] - ordered_vps[1] if len(ordered_vps) > 1 else None
    tile, victim = action.value
    leader_count = sum(a.value[1] == leader for a in legal) if leader is not None else 0
    alternative = any(a.value[1] not in (None, leader) for a in legal) if leader is not None else False
    reachable = leader is not None and leader_count > 0
    return {
        "game_index": game_index,
        "game_seed": game_seed,
        "decision_index": len(game.state.action_records) + 1,
        "completed_turns": game.state.num_turns,
        "agent_index": agent_colors.index(action.color),
        "actor_color": action.color.value,
        "actor_seating_position": order.index(action.color),
        "turn_order": json_cell([color.value for color in order]),
        "victim_color": victim.value if victim is not None else "",
        "victim_relative_offset": (
            (order.index(victim) - order.index(action.color)) % len(order)
            if victim is not None else ""
        ),
        "selected_tile": json_cell(tile),
        "visible_vps": json_cell({color.value: vp for color, vp in visible.items()}),
        "overall_leaders": json_cell([color.value for color, vp in visible.items() if vp == max(visible.values())]),
        "opponent_leaders": json_cell([color.value for color in leaders]),
        "unique_opponent_leader": leader.value if leader is not None else "",
        "highest_opponent_vp": highest,
        "opponent_vp_gap": gap,
        "opponent_leader_tied": leader is None,
        "leader_reachable": reachable,
        "leader_targeting_forced": reachable and leader_count == len(legal),
        "alternative_opponent_reachable": alternative,
        "informative_victim_choice": reachable and alternative,
        "targeted_unique_leader": victim == leader if leader is not None else "",
        "legal_robber_action_count": len(legal),
        "legal_leader_action_count": leader_count if leader is not None else "",
        "random_legal_action_leader_probability": leader_count / len(legal) if leader is not None else "",
        "legal_robber_actions": json_cell([
            {"tile": a.value[0], "victim_color": a.value[1].value if a.value[1] is not None else None}
            for a in legal
        ]),
    }


class RobberDecisionLogger(GameObserver):
    """Capture before execution, then write only after execution succeeds."""

    def __init__(self, writer, *, game_index, game_seed, agent_colors):
        self.writer = writer
        self.game_index = game_index
        self.game_seed = game_seed
        self.agent_colors = agent_colors
        self.pending = None
        self.count = 0

    def step(self, game, action):
        self.pending = robber_decision_row(
            game, action, game_index=self.game_index, game_seed=self.game_seed,
            agent_colors=self.agent_colors,
        )

    def commit(self):
        if self.pending is not None:
            self.writer.writerow(self.pending)
            self.count += 1
            self.pending = None


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(checkpoint, games, seed, output_dir, turn_limit=1000):
    if games < 1 or turn_limit < 1 or seed < 0:
        raise ValueError("games and turn_limit must be positive; seed must be nonnegative")
    checkpoint = Path(checkpoint).resolve()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Choose a new output directory; {output_dir} already exists")

    config = replace(CONFIG, turn_limit=turn_limit)
    core = CatanCore(config)
    torch.set_num_threads(1)
    net, iteration = load_policy_net(checkpoint, core)
    policies = [make_policy(net, i, core.n_agents, deterministic=True) for i in range(core.n_agents)]
    metadata = {
        "schema_version": 1,
        "status": "incomplete",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_iteration": int(iteration),
        "script_sha256": file_sha256(__file__),
        "config": asdict(config),
        "policy_mode": "greedy masked actions; all four checkpoint policies in self-play",
        "agent_colors": [color.value for color in core.colors],
        "games_requested": games,
        "first_game_seed": seed,
        "last_game_seed": seed + games - 1,
        "leader_definition": "Unique highest visible VP among the actor's opponents before execution; actor may lead overall.",
        "baseline_definition": "Uniform over all legal MOVE_ROBBER actions, including actions without a victim.",
        "informative_victim_choice_definition": "Unique opposing leader is reachable and at least one different opponent can also be robbed.",
        "missing_values": "Empty cells mean not applicable. Tied opponents have no unique leader or associated baseline.",
        "indexing": "game_index and decision_index are 1-based; agent_index and seating positions are 0-based.",
        "packages": {name: version(name) for name in ("torch", "torchrl", "tensordict", "catanatron")},
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    total_robber_decisions = 0
    with (output_dir / "robber_decisions.csv").open("w", newline="", encoding="utf-8") as decisions_file, \
            (output_dir / "games.csv").open("w", newline="", encoding="utf-8") as games_file:
        decisions_writer = csv.DictWriter(decisions_file, fieldnames=DECISION_FIELDS)
        games_writer = csv.DictWriter(games_file, fieldnames=GAME_FIELDS)
        decisions_writer.writeheader()
        games_writer.writeheader()
        for index in range(games):
            game_seed = seed + index
            players = [PolicyPlayer(color, core, policy) for color, policy in zip(core.colors, policies)]
            game = core.new_game(game_seed, players=players)
            logger = RobberDecisionLogger(
                decisions_writer, game_index=index + 1, game_seed=game_seed,
                agent_colors=core.colors,
            )
            game.watch(logger)
            while not core.is_over(game):
                game.play_tick()
                logger.commit()
            winner = game.winning_color()
            terminated, truncated = core.status(game)
            games_writer.writerow({
                "game_index": index + 1,
                "game_seed": game_seed,
                "turn_order": json_cell([color.value for color in game.state.colors]),
                "winner_color": winner.value if winner is not None else "",
                "terminated": terminated,
                "truncated": truncated,
                "completed_turns": game.state.num_turns,
                "decision_count": len(game.state.action_records),
                "robber_decision_count": logger.count,
                "final_visible_vps": json_cell({color.value: get_visible_victory_points(game.state, color) for color in core.colors}),
            })
            decisions_file.flush()
            games_file.flush()
            total_robber_decisions += logger.count
            print(f"Game {index + 1}/{games}: {logger.count} robber decisions, "
                  f"{game.state.num_turns} turns, winner={winner.value if winner else 'none (turn limit)'}", flush=True)

    metadata.update(status="complete", games_completed=games, robber_decisions=total_robber_decisions)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Logged {total_robber_decisions} robber decisions to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0, help="First game seed; later games use consecutive seeds.")
    parser.add_argument("--output-dir", type=Path, default=Path("logs/rq1"), help="New directory for CSVs and metadata; existing directories are not overwritten.")
    parser.add_argument("--turn-limit", type=int, default=1000)
    args = parser.parse_args()
    try:
        run(args.checkpoint, args.games, args.seed, args.output_dir, args.turn_limit)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
