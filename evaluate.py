"""Play policies against Catanatron's bots (or each other): win rates, robber and trade stats.

    result = play_match([policy, "random", "value", "value"], n_games=100)
    print(result.table())

runeval.py does this for trained checkpoints. A policy is any function  policy(obs [D], legal_mask [A]) -> action index,
using the same observations and action indices as CatanEnv. Wrap a trained
network in one to evaluate it. With n players, chance win rate is 1/n.

With CatanConfig(trading=True) policies can offer trades. Catanatron's bots never
offer. Asked about an offer they could accept, random / weighted / victory_point
say yes about half the time, value always says no, and alphabeta cannot answer.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import torch
import numpy as np
from catanatron.models.player import Player, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.models.enums import ActionType
from catanatron.state_functions import get_actual_victory_points, get_player_freqdeck

from catan_env import CatanConfig, CatanCore, IllegalActionError, RobberStats, TradeStats
from rewardfn import BUILDS, build_limits, hand_utility

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

class TradeQuality(TradeStats):
    """TradeStats plus how sensible the trading was, judged by rewardfn.hand_utility. Per color:

      helped / hurt            its completed trades that raised / lowered its hand utility
      built_after              its completed trades after which it built something that uses a
                               card it received, before its next end of turn
      asked_good / yes_good    offers that would have raised its hand utility: times asked / accepted
      asked_bad / yes_bad      the same for offers that would have lowered it
    """

    FIELDS = TradeStats.FIELDS + ("helped", "hurt", "built_after", "asked_good", "yes_good", "asked_bad", "yes_bad")
    BUILD_ACTIONS = {ActionType.BUILD_SETTLEMENT: "settlement", ActionType.BUILD_CITY: "city",
                     ActionType.BUILD_ROAD: "road"}

    def before(self, game):
        super().before(game)
        self.unused = defaultdict(list)  # color -> for each trade not built with yet: which cards it received

    @staticmethod
    def gain(game, color, received) -> float:
        """Change in `color`'s hand utility if it received these cards now (negative counts = given away)."""
        hand, limits = np.array(get_player_freqdeck(game.state, color)), build_limits(game, color)
        return hand_utility(hand + received, limits) - hand_utility(hand, limits)

    def step(self, game, action):  # called before the action is played
        super().step(game, action)
        kind, me = action.action_type, self.by_color[action.color]
        if kind in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE, ActionType.CONFIRM_TRADE):
            gives, asks = np.array(action.value[:5]), np.array(action.value[5:10])  # from the offerer's side
        if kind in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE):
            gain = self.gain(game, action.color, gives - asks)
            if abs(gain) > 1e-9:
                side = "good" if gain > 0 else "bad"
                me[f"asked_{side}"] += 1
                me[f"yes_{side}"] += int(kind == ActionType.ACCEPT_TRADE)
        elif kind == ActionType.CONFIRM_TRADE:
            for color, received in ((action.color, asks - gives), (action.value[10], gives - asks)):
                gain = self.gain(game, color, received)
                self.by_color[color]["helped"] += int(gain > 1e-9)
                self.by_color[color]["hurt"] += int(gain < -1e-9)
                self.unused[color].append(received > 0)
        elif kind in self.BUILD_ACTIONS:
            in_cost = BUILDS[self.BUILD_ACTIONS[kind]][0] > 0
            used = [bool((received & in_cost).any()) for received in self.unused[action.color]]
            me["built_after"] += sum(used)
            self.unused[action.color] = [r for r, u in zip(self.unused[action.color], used) if not u]
        elif kind == ActionType.END_TURN:
            self.unused[action.color] = []


@dataclass
class MatchResult:
    labels: list
    n_games: int
    wins: np.ndarray
    truncated: int
    mean_turns: float
    mean_final_vps: np.ndarray
    robber: list  # per seat: summed RobberStats fields
    trade: Optional[list] = None  # per seat: summed TradeQuality fields

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
        if self.trade and any(t["offers"] for t in self.trade):
            lines += ["", self.trade_table()]
        return "\n".join(lines)

    def trade_table(self) -> str:
        share = lambda part, whole: f"{part / whole:.0%}" if whole else "-"
        lines = [f"{'seat':<14}{'offers':>7}{'trades':>7}{'helped':>8}{'hurt':>6}{'built after':>13}"
                 f"{'yes if good':>13}{'yes if bad':>12}{'with leader':>13}"]
        for label, t in zip(self.labels, self.trade):
            lines.append(f"{label:<14}{t['offers']:>7}{t['trades']:>7}{share(t['helped'], t['trades']):>8}"
                         f"{share(t['hurt'], t['trades']):>6}{share(t['built_after'], t['trades']):>13}"
                         f"{share(t['yes_good'], t['asked_good']):>13}{share(t['yes_bad'], t['asked_bad']):>12}"
                         f"{share(t['trades_with_leader'], t['trades']):>13}")
        lines.append("trade columns are explained in TradeQuality (evaluate.py)")
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
    trade = [dict.fromkeys(TradeQuality.FIELDS, 0) for _ in range(n)]
    for g in range(n_games):
        players = [BOTS[s](c) if isinstance(s, str) else PolicyPlayer(c, core, s) for s, c in zip(seats, core.colors)]
        game = core.new_game(seed + g, players=players)
        stats, trade_stats = RobberStats(), TradeQuality()
        game.watch(stats)
        game.watch(trade_stats)
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
            for k, v in trade_stats.summary().get(color.value, {}).items():
                trade[i][k] += v

    return MatchResult(labels, n_games, wins, truncated, float(np.mean(turns)), final_vps / n_games, robber, trade)

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
