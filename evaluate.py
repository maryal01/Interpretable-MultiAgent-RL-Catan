"""Play policies against Catanatron's bots (or each other): win rates and robber stats.

    python evaluate.py --opponents random weighted value --games 100

A policy is any function  policy(obs [D], legal_mask [A]) -> action index,
using the same observations and action indices as CatanEnv. Wrap a trained
network in one to evaluate it. With n players, chance win rate is 1/n.

With CatanConfig(trading=True) policies can offer trades. Catanatron's bots never
offer. Asked about an offer they could accept, random / weighted / victory_point
say yes about half the time, value always says no, and alphabeta cannot answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import torch
import numpy as np
import argparse
from catanatron.models.player import Player, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.state_functions import get_actual_victory_points

from catan_env import CatanConfig, CatanCore, IllegalActionError, RobberStats

Policy = Callable[[np.ndarray, np.ndarray], int]

# Roughly weakest to strongest. AlphaBeta searches game trees and is much slower.
BOTS = {
    "random": RandomPlayer,
    "weighted": WeightedRandomPlayer,
    "victory_point": VictoryPointPlayer,
    "value": ValueFunctionPlayer,
    "alphabeta": AlphaBetaPlayer,
}

def random_policy(seed: Optional[int] = None) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda obs, mask: int(rng.choice(np.flatnonzero(mask)))

class PolicyPlayer(Player):
    """Lets Catanatron drive a game while a Policy makes this seat's decisions."""

    def __init__(self, color, core: CatanCore, policy: Policy):
        super().__init__(color)
        self.core, self.policy, self.agent = core, policy, core.colors.index(color)

    def decide(self, game, playable_actions):
        obs = self.core.observation(game, self.agent)
        mask = self.core.legal_action_mask(game, self.agent)
        index = int(self.policy(obs, mask))
        action = self.core.index_to_action(game, index, self.agent)
        if not mask[index]:  # the mask, not playable_actions: the engine never lists trade offers
            raise IllegalActionError(f"policy chose {action}, which is not legal")
        return action

@dataclass
class MatchResult:
    labels: list
    n_games: int
    wins: np.ndarray
    truncated: int
    mean_turns: float
    mean_final_vps: np.ndarray
    robber: list  # per seat: summed RobberStats fields

    @property
    def win_rates(self) -> np.ndarray:
        return self.wins / self.n_games

    def table(self) -> str:
        lines = [f"{self.n_games} games, {self.truncated} hit the turn limit, mean {self.mean_turns:.0f} turns",
                 f"{'seat':<14}{'win rate':>9}{'final VP':>10}{'steals*':>9}{'from leader':>13}{'by chance':>11}"]
        for i, label in enumerate(self.labels):
            r = self.robber[i]
            lines.append(f"{label:<14}{self.win_rates[i]:>9.2f}{self.mean_final_vps[i]:>10.2f}"
                         f"{r['clear_leader_steals']:>9}{r['steals_from_leader']:>13}{r['chance_from_leader']:>11.1f}")
        lines.append("* steals made while one opponent clearly led; 'by chance' = expected if stealing at random")
        return "\n".join(lines)

def play_match(seats: Sequence, n_games: int = 100, config: CatanConfig = CatanConfig(),
               seed: int = 0, labels: Optional[Sequence[str]] = None) -> MatchResult:
    """seats[i] is a bot name from BOTS or a Policy; it plays color i. Turn order is shuffled each game."""
    core = CatanCore(config)
    n = core.n_agents
    if len(seats) != n:
        raise ValueError(f"need {n} seats for a {n}-player game, got {len(seats)}")
    for s in seats:
        if isinstance(s, str) and s not in BOTS:
            raise ValueError(f"unknown bot {s!r}; choose from {sorted(BOTS)}")
    if config.trading and "alphabeta" in seats:
        raise ValueError("alphabeta cannot answer trade offers; choose another bot when trading is on")
    labels = list(labels or [s if isinstance(s, str) else f"policy_{i}" for i, s in enumerate(seats)])

    wins, final_vps = np.zeros(n, dtype=int), np.zeros(n)
    turns, truncated = [], 0
    robber = [dict.fromkeys(RobberStats.FIELDS, 0) for _ in range(n)]
    for g in range(n_games):
        players = [BOTS[s](c) if isinstance(s, str) else PolicyPlayer(c, core, s) for s, c in zip(seats, core.colors)]
        game = core.new_game(seed + g, players=players)
        stats = RobberStats()
        game.watch(stats)
        while not core.is_over(game):
            game.play_tick()
            core.skip_forced_rejects(game)  # trade offers are answered as in training
        winner = game.winning_color()
        if winner is None:
            truncated += 1
        else:
            wins[core.colors.index(winner)] += 1
        turns.append(game.state.num_turns)
        for i, color in enumerate(core.colors):
            final_vps[i] += get_actual_victory_points(game.state, color)
            for k, v in stats.summary().get(color.value, {}).items():
                robber[i][k] += v

    return MatchResult(labels, n_games, wins, truncated, float(np.mean(turns)), final_vps / n_games, robber)

def make_policy(policy_net, seat, n_agents, deterministic=True):
    def policy(obs, mask):
        x = torch.zeros(n_agents, obs.shape[0])
        x[seat] = torch.as_tensor(obs)
        with torch.no_grad():
            logits = policy_net(x)[seat]
        logits = logits.masked_fill(~torch.as_tensor(mask), float("-inf"))
        if deterministic:
            return int(logits.argmax())
        return int(torch.distributions.Categorical(logits=logits).sample())
    return policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponents", nargs="+", default=["random", "random", "random"], choices=sorted(BOTS))
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--map", default="BASE", choices=["MINI", "BASE", "TOURNAMENT"])
    parser.add_argument("--vps", type=int, default=8)
    parser.add_argument("--trading", action="store_true")
    args = parser.parse_args()

    config = CatanConfig(num_players=1 + len(args.opponents), map_type=args.map, vps_to_win=args.vps, trading=args.trading)
    policy = random_policy(args.seed)  # TODO: replace with your trained policy
    print(play_match([policy, *args.opponents], args.games, config, args.seed,
                     labels=["our policy", *args.opponents]).table())
