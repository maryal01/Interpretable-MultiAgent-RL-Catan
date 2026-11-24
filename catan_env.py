"""Multi-agent Catan for TorchRL, built on Catanatron's rules engine.

Every player at the table is an agent you train. All players sit in one TorchRL
group, "agents", so ``MultiAgentMLP(share_params=True/False)`` switches between
one shared network (self-play) and an independent network per player.

    env = CatanEnv(CatanConfig(num_players=4, map_type="BASE", vps_to_win=8), seed=0)
    td = env.reset()

Keys (n players, A actions, D observation size):
    OBS          ("agents", "observation")  float [n, D]  each player's own view
    ACTION_MASK  ("agents", "action_mask")  bool  [n, A]  legal moves; rows of players not acting are all True
    ACTIVE       ("agents", "mask")         bool  [n]     True for whose turn it is
    ACTION       ("agents", "action")       int   [n]     only the active entry is used
    REWARD       ("agents", "reward")       float [n, 1]  under "next"; +1 win / -1 loss at the end
    "done" [1] and ("agents", "done") [n, 1]

Design notes
  * Catanatron ``Player`` objects are placeholders: they only define the seats
    (colors). Decisions come from your agents via ``env.step``. Catanatron
    shuffles turn order every game.
  * Robber victims are encoded relative to the actor ("rob the next player"),
    so an action index means the same thing whichever seat an agent is in.
  * ``CatanEnv(bots=("value",))`` puts Catanatron bots at the table: each game they
    take random seats and play their moves inside ``env.step``; the other seats
    are agents. A bot seat is never active, so it adds no training data.
  * Parallelise with a multi-process ``Collector`` (one env per worker), not
    ``ParallelEnv``, which syncs every step and was slower than one process.

Player-to-player trading (``CatanConfig(trading=True)``, off by default)
  * Offers: give 1 or 2 of one resource for 1 of another (40 offers). An offer is
    legal only on your own turn after rolling, only if you hold the cards (the
    engine does not check this), only if some opponent holds the asked card (no
    offers that nobody can accept; this tells the offerer "someone has ore" and
    nothing more), and at most ``max_offers_per_turn`` times per turn (0 = no
    offers: learn the base game first, then raise it without changing shapes).
  * Responses: each opponent who holds the asked card, in seat order, accepts or
    rejects; the others are skipped, so nobody is asked a question with only one
    answer. If anyone accepted, the offerer confirms with one of them (relative
    seat, like the robber) or cancels. While a trade is being answered, the
    player being asked is the active agent.
  * Observations get extra trade features (offer, who made it, who accepted).
  * With trading off, actions and observations are exactly as before, so older
    checkpoints still load.

Run ``python catan_env.py --workers 4`` to print the specs and benchmark speed.
"""

from __future__ import annotations

import operator
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from catanatron.features import create_sample, get_feature_ordering
from catanatron.game import Game
from catanatron.gym.envs.action_space import get_action_array
from catanatron.models.enums import RESOURCES, Action, ActionPrompt, ActionType
from catanatron.models.map import build_map
from catanatron.models.player import Color, Player, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron.observer import GameObserver
from catanatron.state_functions import get_player_freqdeck, get_visible_victory_points, player_has_rolled

GROUP = "agents"
OBS = (GROUP, "observation")
ACTION = (GROUP, "action")
ACTION_MASK = (GROUP, "action_mask")
ACTIVE = (GROUP, "mask")
REWARD = (GROUP, "reward")
DONE = (GROUP, "done")

ALL_COLORS = (Color.RED, Color.BLUE, Color.ORANGE, Color.WHITE)

# Catanatron's bots, roughly weakest to strongest. AlphaBeta searches game trees and is much slower.
BOTS = {
    "random": RandomPlayer,
    "weighted": WeightedRandomPlayer,
    "victory_point": VictoryPointPlayer,
    "value": ValueFunctionPlayer,
    "alphabeta": AlphaBetaPlayer,
}

class IllegalActionError(ValueError):
    pass


@dataclass(frozen=True)
class CatanConfig:
    num_players: int = 4
    map_type: str = "BASE"  # "MINI" (7 tiles), "BASE" (19 tiles) or "TOURNAMENT"
    vps_to_win: int = 8
    turn_limit: int = 1000  # no winner after this many turns -> truncated
    dev_cards: bool = False
    trading: bool = False  # player-to-player trades
    max_offers_per_turn: int = 1  # only used when trading=True; 0 = nobody may offer

    def __post_init__(self):
        if not 2 <= self.num_players <= 4:
            raise ValueError("num_players must be 2, 3 or 4")
        if self.map_type not in ("MINI", "BASE", "TOURNAMENT"):
            raise ValueError(f"unknown map_type {self.map_type!r}")
        if self.max_offers_per_turn < 0:
            raise ValueError("max_offers_per_turn must be 0 or more")


def trade_offers() -> list[tuple]:
    """The offers agents can make, in Catanatron's format: 5 counts given, then 5 asked
    (wood, brick, sheep, wheat, ore). Give 1 or 2 of one resource for 1 of another."""
    offers = []
    for give in range(5):
        for ask in range(5):
            if give == ask:
                continue
            for n_give in (1, 2):
                value = [0] * 10
                value[give] = n_give
                value[5 + ask] = 1
                offers.append(tuple(value))
    return offers


# ----------------------------------------------------------------- rewards
class RewardFn(Protocol):
    def reset(self, core: "CatanCore", game: Game) -> None: ...
    def __call__(self, core: "CatanCore", game: Game) -> np.ndarray: ...  # [n] rewards for this step


class WinLossReward:
    """+1 winner, -1 everyone else when the game ends; 0 otherwise (and on truncation)."""

    def reset(self, core, game):
        pass

    def __call__(self, core, game):
        rewards = np.zeros(core.n_agents, dtype=np.float32)
        winner = game.winning_color()
        if winner is not None:
            rewards[:] = -1.0
            rewards[core.colors.index(winner)] = 1.0
        return rewards


# -------------------------------------------------------------------- core
class CatanCore:
    """All Catan-specific logic, on a plain Catanatron Game. Used by the env and by evaluate.py."""

    def __init__(self, config: CatanConfig = CatanConfig(), reward_fn: Optional[RewardFn] = None):
        self.config = config
        self.n_agents = config.num_players
        self.colors = ALL_COLORS[: self.n_agents]  # agent i always plays colors[i]
        self.reward_fn = reward_fn or WinLossReward()
        self.action_table = self._build_action_table()
        self._action_index = {a: i for i, a in enumerate(self.action_table)}
        self.n_actions = len(self.action_table)
        base_features = list(get_feature_ordering(self.n_agents, config.map_type))
        self._pick_features = operator.itemgetter(*base_features)
        self.feature_names = base_features + (self._trade_feature_names() if config.trading else [])
        self.obs_dim = len(self.feature_names)
        if config.trading:
            offers = [(i, v) for i, (t, v) in enumerate(self.action_table) if t == ActionType.OFFER_TRADE]
            self._offer_indices = np.array([i for i, _ in offers])
            self._offer_gives = np.array([v[:5] for _, v in offers])  # [n_offers, 5]
            self._offer_asks = np.array([v[5:].index(1) for _, v in offers])  # [n_offers] asked resource

    def _build_action_table(self) -> list[tuple]:
        # Catanatron's Gym action list. Robber victims (and trade partners) use
        # self.colors as stand-ins for "me, next player, ...".
        table = list(get_action_array(self.colors, self.config.map_type))
        if self.config.trading:
            table += [(ActionType.OFFER_TRADE, offer) for offer in trade_offers()]
            # Accept/reject carry the whole offer in Catanatron; here one index each.
            table += [(ActionType.ACCEPT_TRADE, None), (ActionType.REJECT_TRADE, None),
                      (ActionType.CANCEL_TRADE, None)]
            # "Confirm with the player k seats after me", k = 1 .. n-1.
            table += [(ActionType.CONFIRM_TRADE, self.colors[k]) for k in range(1, self.n_agents)]
        return table

    def _trade_feature_names(self) -> list[str]:
        names = ["TRADE_IN_PROGRESS", "TRADE_I_AM_ASKED", "TRADE_I_CHOOSE_PARTNER"]
        names += [f"TRADE_OFFER_GIVES_{r}" for r in RESOURCES]  # from the offerer's side
        names += [f"TRADE_OFFER_ASKS_{r}" for r in RESOURCES]
        names += [f"TRADE_OFFERER_P{k}" for k in range(self.n_agents)]  # P0 = me, P1 = next, ...
        names += [f"TRADE_ACCEPTED_P{k}" for k in range(self.n_agents)]
        return names

    def new_game(self, seed: Optional[int] = None, players: Optional[list] = None) -> Game:
        """Same seed -> same map, turn order and dice. `players` only for bot matches."""
        players = players or [Player(c) for c in self.colors]  # placeholders: seats only
        if [p.color for p in players] != list(self.colors):
            raise ValueError(f"players must have colors {[c.value for c in self.colors]} in that order")
        if seed is not None:
            random.seed(seed)  # Catanatron builds the map with the global RNG
        game = Game(players=players, seed=seed, catan_map=build_map(self.config.map_type),
                    vps_to_win=self.config.vps_to_win)
        
        if self.config.dev_cards == False:
            game.state.development_listdeck = []

        self.reward_fn.reset(self, game)
        return game

    def current_agent(self, game: Game) -> int:
        return self.colors.index(game.state.current_color())

    def status(self, game: Game) -> tuple[bool, bool]:
        """(terminated, truncated)"""
        terminated = game.winning_color() is not None
        return terminated, (not terminated) and game.state.num_turns >= self.config.turn_limit

    def is_over(self, game: Game) -> bool:
        return any(self.status(game))

    def rewards(self, game: Game) -> np.ndarray:
        return self.reward_fn(self, game)

    def observation(self, game: Game, agent: int) -> np.ndarray:
        """Catanatron's features from this agent's point of view (opponents' hands hidden),
        plus trade features when trading is on."""
        base = np.array(self._pick_features(create_sample(game, self.colors[agent])), dtype=np.float32)
        if not self.config.trading:
            return base
        return np.concatenate([base, self._trade_features(game, agent)])

    def _trade_features(self, game: Game, agent: int) -> np.ndarray:
        state, me, n = game.state, self.colors[agent], self.n_agents
        f = np.zeros(13 + 2 * n, dtype=np.float32)
        if not state.is_resolving_trade:
            return f
        asking_me = state.current_color() == me
        f[0] = 1.0
        f[1] = state.current_prompt == ActionPrompt.DECIDE_TRADE and asking_me
        f[2] = state.current_prompt == ActionPrompt.DECIDE_ACCEPTEES and asking_me
        f[3:13] = state.current_trade[:10]
        offerer = state.colors[state.current_turn_index]
        f[13 + self._relative(game, me, offerer)] = 1.0
        for color, accepted in zip(state.colors, state.acceptees):
            if accepted:
                f[13 + n + self._relative(game, me, color)] = 1.0
        return f

    def legal_action_mask(self, game: Game, agent: int) -> np.ndarray:
        """Legal moves for `agent`; all False if it isn't their turn."""
        mask = np.zeros(self.n_actions, dtype=bool)
        if self.is_over(game) or self.current_agent(game) != agent:
            return mask
        for action in game.playable_actions:
            mask[self.action_to_index(game, action)] = True
        if self.config.trading and self._can_offer(game):
            state, me = game.state, self.colors[agent]
            hand = np.array(get_player_freqdeck(state, me))
            others = np.array([get_player_freqdeck(state, c) for c in self.colors if c != me])
            mask[self._offer_indices] = ((hand >= self._offer_gives).all(axis=1)  # I hold what I give
                                         & others.any(axis=0)[self._offer_asks])  # someone holds what I ask
        return mask

    def _can_offer(self, game: Game) -> bool:
        state = game.state
        return (state.current_prompt == ActionPrompt.PLAY_TURN
                and not state.is_resolving_trade
                and player_has_rolled(state, state.current_color())
                and self.offers_this_turn(game) < self.config.max_offers_per_turn)

    def offers_this_turn(self, game: Game) -> int:
        """Offers the current turn's player has made since rolling the dice this turn."""
        color = game.state.colors[game.state.current_turn_index]
        count = 0
        for record in reversed(game.state.action_records):
            action = record.action
            if action.color == color and action.action_type == ActionType.ROLL:
                break
            if action.color == color and action.action_type == ActionType.OFFER_TRADE:
                count += 1
        return count

    def legal_indices(self, game: Game) -> np.ndarray:
        return np.flatnonzero(self.legal_action_mask(game, self.current_agent(game)))

    def _relative(self, game: Game, actor: Color, other: Color) -> int:
        order = game.state.colors  # turn order
        return (order.index(other) - order.index(actor)) % self.n_agents

    def _absolute(self, game: Game, actor: Color, rel: int) -> Color:
        order = game.state.colors
        return order[(order.index(actor) + rel) % self.n_agents]

    def action_to_index(self, game: Game, action: Action) -> int:
        action_type, value = action.action_type, action.value
        if action_type == ActionType.MOVE_ROBBER and value[1] is not None:
            value = (value[0], self.colors[self._relative(game, action.color, value[1])])
        elif action_type in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE):
            value = None
        elif action_type == ActionType.CONFIRM_TRADE:
            value = self.colors[self._relative(game, action.color, value[10])]
        elif action_type == ActionType.OFFER_TRADE:
            value = tuple(value[:10])
        return self._action_index[(action_type, value)]

    def index_to_action(self, game: Game, index: int, agent: int) -> Action:
        actor = self.colors[agent]
        action_type, value = self.action_table[index]
        if action_type == ActionType.MOVE_ROBBER and value[1] is not None:
            value = (value[0], self._absolute(game, actor, self.colors.index(value[1])))
        elif action_type in (ActionType.ACCEPT_TRADE, ActionType.REJECT_TRADE):
            value = game.state.current_trade
        elif action_type == ActionType.CONFIRM_TRADE:
            partner = self._absolute(game, actor, self.colors.index(value))
            value = (*game.state.current_trade[:10], partner)
        return Action(actor, action_type, value)

    def apply(self, game: Game, agent: int, index: int, observers: Iterable = ()) -> Action:
        """Play `agent`'s action `index`. Raises IllegalActionError if it isn't legal now."""
        if self.is_over(game):
            raise IllegalActionError("the game is already over")
        if agent != self.current_agent(game):
            raise IllegalActionError(f"it is not player_{agent}'s turn")
        if not 0 <= index < self.n_actions:
            raise IllegalActionError(f"action index {index} out of range")
        action = self.index_to_action(game, index, agent)
        if not self.legal_action_mask(game, agent)[index]:  # covers offers, which the engine doesn't list
            raise IllegalActionError(f"{action} is not legal right now")
        for observer in observers:
            observer.step(game, action)  # observers see the game before the action
        game.execute(action)
        self.skip_forced_rejects(game)
        return action

    @staticmethod
    def skip_forced_rejects(game: Game) -> None:
        """Answer "reject" for players who have no real choice, so the question moves on:
          * players who don't hold the asked cards (reject is their only legal move), and
          * the offerer: a Catanatron bug asks them about their own offer whenever they
            don't sit first.
        No-op when trading is off."""
        state = game.state
        while (state.is_resolving_trade and state.current_prompt == ActionPrompt.DECIDE_TRADE
               and (len(game.playable_actions) == 1
                    or state.current_color() == state.colors[state.current_turn_index])):
            game.execute(Action(state.current_color(), ActionType.REJECT_TRADE, state.current_trade))


# ------------------------------------------------------------------- stats
class RobberStats(GameObserver):
    """Does a player rob the leader more than chance would? Counts per color for one game.

    Only robber steals made while the player's opponents had a single clear leader (most visible VPs, no tie) count:
      clear_leader_steals   steals made in that situation
      steals_from_leader    ... where the victim was the leader
      chance_from_leader    expected hits on the leader if stealing at random
    Deliberate targeting shows up as steals_from_leader > chance_from_leader.
    """

    FIELDS = ("steals", "clear_leader_steals", "steals_from_leader", "chance_from_leader")

    def __init__(self, *args, **kwargs):
        self.by_color = defaultdict(lambda: dict.fromkeys(self.FIELDS, 0))

    def before(self, game):
        self.by_color.clear()

    def step(self, game, action):
        if action.action_type != ActionType.MOVE_ROBBER or action.value[1] is None:
            return
        record = self.by_color[action.color]
        record["steals"] += 1
        vps = {c: get_visible_victory_points(game.state, c) for c in game.state.colors if c != action.color}
        leaders = [c for c, v in vps.items() if v == max(vps.values())]
        if len(leaders) != 1:
            return
        options = [a.value[1] for a in game.playable_actions
                   if a.action_type == ActionType.MOVE_ROBBER and a.value[1] is not None]
        record["clear_leader_steals"] += 1
        record["steals_from_leader"] += int(action.value[1] == leaders[0])
        record["chance_from_leader"] += sum(v == leaders[0] for v in options) / len(options)

    def summary(self) -> dict:
        return {color.value: dict(record) for color, record in self.by_color.items()}


class TradeStats(GameObserver):
    """Trading activity per color for one game (all zeros when trading is off).

      offers              offers this player made
      accepts / rejects   how this player answered offers it could have accepted
      cancels             offers this player called off after someone accepted
      trades              completed trades (counted for both sides)
      trades_with_leader  completed trades whose partner was this player's single clear
                          leader (most visible VPs among its opponents) at that moment
    """

    FIELDS = ("offers", "accepts", "rejects", "cancels", "trades", "trades_with_leader")

    def __init__(self, *args, **kwargs):
        self.by_color = defaultdict(lambda: dict.fromkeys(self.FIELDS, 0))

    def before(self, game):
        self.by_color.clear()

    def step(self, game, action):
        field = {
            ActionType.OFFER_TRADE: "offers", 
            ActionType.ACCEPT_TRADE: "accepts",
            ActionType.REJECT_TRADE: "rejects", 
            ActionType.CANCEL_TRADE: "cancels"
        }.get(action.action_type)

        if field:
            self.by_color[action.color][field] += 1
        elif action.action_type == ActionType.CONFIRM_TRADE:
            partner = action.value[10]
            for me, other in ((action.color, partner), (partner, action.color)):
                self.by_color[me]["trades"] += 1
                vps = {c: get_visible_victory_points(game.state, c) for c in game.state.colors if c != me}
                leaders = [c for c, v in vps.items() if v == max(vps.values())]
                if leaders == [other]:
                    self.by_color[me]["trades_with_leader"] += 1

    def summary(self) -> dict:
        return { color.value: dict(record) for color, record in self.by_color.items() }


# --------------------------------------------------------------------- env
class CatanEnv(EnvBase):
    def __init__(self, config: CatanConfig = CatanConfig(), reward_fn: Optional[RewardFn] = None,
                 seed: Optional[int] = None, device="cpu", update_spec_mask: bool = True,
                 bots: Sequence[str] = ()):
        """update_spec_mask: keep the action spec's mask in sync with legal moves so
        env.rand_action() / env.rollout() without a policy stay legal.
        bots: names from BOTS that join every game at random seats (fewer than num_players)."""
        super().__init__(device=device, batch_size=torch.Size([]))
        self.core = CatanCore(config, reward_fn)
        for b in bots:
            if b not in BOTS:
                raise ValueError(f"unknown bot {b!r}; choose from {sorted(BOTS)}")
        if len(bots) >= self.core.n_agents:
            raise ValueError("bots must leave at least one seat for an agent")
        if config.trading and "alphabeta" in bots:
            raise ValueError("alphabeta cannot answer trade offers; choose another bot when trading is on")
        self.bots = tuple(bots)
        self.bot_players: dict = {}  # agent index -> Catanatron Player, for the current game
        self.group_map = {GROUP: [f"player_{i}" for i in range(self.core.n_agents)]}
        self.update_spec_mask = update_spec_mask
        self._rng = np.random.default_rng(seed)
        self.game: Optional[Game] = None
        self.last_game_seed: Optional[int] = None
        self.stats = RobberStats()
        self.last_episode_stats: dict = {}
        self.trade_stats = TradeStats()
        self.last_episode_trade_stats: dict = {}
        self._make_specs()

    def _make_specs(self):
        n, A, D = self.core.n_agents, self.core.n_actions, self.core.obs_dim
        flag = lambda shape: Categorical(n=2, shape=shape, dtype=torch.bool, device=self.device)  # noqa: E731
        self.full_observation_spec = Composite({GROUP: Composite({
            "observation": Unbounded(shape=(n, D), dtype=torch.float32, device=self.device),
            "action_mask": flag((n, A)),
            "mask": flag((n,)),
        }, shape=(n,))})
        self.full_action_spec = Composite({GROUP: Composite({
            "action": Categorical(n=A, shape=(n,), dtype=torch.int64, device=self.device)}, shape=(n,))})
        self.full_reward_spec = Composite({GROUP: Composite({
            "reward": Unbounded(shape=(n, 1), dtype=torch.float32, device=self.device)}, shape=(n,))})
        self.full_done_spec = Composite({
            "done": flag((1,)), "terminated": flag((1,)), "truncated": flag((1,)),
            GROUP: Composite({"done": flag((n, 1)), "terminated": flag((n, 1)), "truncated": flag((n, 1))},
                             shape=(n,)),
        })

    def _observe(self, terminated: bool, truncated: bool) -> TensorDict:
        core, n = self.core, self.core.n_agents
        obs = np.stack([core.observation(self.game, i) for i in range(n)])
        action_mask = np.ones((n, core.n_actions), dtype=bool)
        active = np.zeros(n, dtype=bool)
        if not (terminated or truncated):
            i = core.current_agent(self.game)
            active[i] = True
            action_mask[i] = core.legal_action_mask(self.game, i)
        action_mask = torch.from_numpy(action_mask)
        if self.update_spec_mask:
            self.full_action_spec[ACTION].update_mask(action_mask.clone())
        done = terminated or truncated
        return TensorDict({
            GROUP: TensorDict({
                "observation": torch.from_numpy(obs),
                "action_mask": action_mask,
                "mask": torch.from_numpy(active),
                "done": torch.full((n, 1), done),
                "terminated": torch.full((n, 1), terminated),
                "truncated": torch.full((n, 1), truncated),
            }, batch_size=[n]),
            "done": torch.tensor([done]),
            "terminated": torch.tensor([terminated]),
            "truncated": torch.tensor([truncated]),
        }, batch_size=[], device=self.device)

    def _reset(self, tensordict: Optional[TensorDictBase] = None, **kwargs) -> TensorDict:
        self.last_game_seed = int(self._rng.integers(2**31 - 1))
        colors = self.core.colors
        seats = self._rng.choice(len(colors), size=len(self.bots), replace=False) if self.bots else []
        self.bot_players = {int(s): BOTS[b](colors[s]) for s, b in zip(seats, self.bots)}
        players = [self.bot_players.get(i) or Player(c) for i, c in enumerate(colors)]
        self.game = self.core.new_game(self.last_game_seed, players=players)
        self.stats.before(self.game)
        self.trade_stats.before(self.game)
        self._play_bots()
        return self._observe(False, False)

    def _play_bots(self) -> None:
        """Play bot moves until an agent has to act or the game ends."""
        game, core = self.game, self.core
        while not core.is_over(game) and core.current_agent(game) in self.bot_players:
            action = self.bot_players[core.current_agent(game)].decide(game, game.playable_actions)
            for observer in (self.stats, self.trade_stats):
                observer.step(game, action)
            game.execute(action)
            core.skip_forced_rejects(game)

    def _step(self, tensordict: TensorDictBase) -> TensorDict:
        agent = self.core.current_agent(self.game)
        self.core.apply(self.game, agent, int(tensordict.get(ACTION)[agent]), observers=(self.stats, self.trade_stats))
        self._play_bots()
        terminated, truncated = self.core.status(self.game)
        if terminated or truncated:
            self.last_episode_stats = self.stats.summary()
            self.last_episode_trade_stats = self.trade_stats.summary()
        out = self._observe(terminated, truncated)
        out.set(REWARD, torch.from_numpy(self.core.rewards(self.game)).unsqueeze(-1).to(self.device))
        return out

    def _set_seed(self, seed: Optional[int]) -> None:
        self._rng = np.random.default_rng(seed)


def make_env(seed: Optional[int] = None, reward_fn: Optional[RewardFn] = None, bots: Sequence[str] = (),
             **config) -> CatanEnv:
    """Picklable factory for collectors: functools.partial(make_env, seed=s, num_players=4)."""
    return CatanEnv(CatanConfig(**config), reward_fn=reward_fn, seed=seed, bots=bots)


# --------------------------------------------------------------- benchmark
def _masked_random_policy():
    from tensordict.nn import TensorDictModule

    def sample(mask):
        return torch.multinomial(mask.float().reshape(-1, mask.shape[-1]), 1).reshape(mask.shape[:-1])

    return TensorDictModule(sample, in_keys=[ACTION_MASK], out_keys=[ACTION])


def _benchmark(workers: int, frames: int, **config):
    import functools
    import time

    from torchrl.collectors import Collector

    torch.set_num_threads(1)
    results = {}
    for w in sorted({1, workers}):
        envs = [functools.partial(make_env, seed=s, **config) for s in range(w)]
        kwargs = dict(frames_per_batch=1000, total_frames=frames + 1000)
        collector = (Collector(envs, _masked_random_policy(), num_collectors=w, sync=True, **kwargs) if w > 1
                     else Collector(envs[0], _masked_random_policy(), **kwargs))
        batches = iter(collector)
        next(batches)  # warm-up
        t0, n = time.perf_counter(), 0
        for batch in batches:
            n += batch.numel()
        results[w] = n / (time.perf_counter() - t0)
        collector.shutdown()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print env specs and benchmark speed (random legal moves).")
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--map", default="BASE", choices=["MINI", "BASE", "TOURNAMENT"])
    parser.add_argument("--vps", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--frames", type=int, default=5000)
    args = parser.parse_args()
    config = dict(num_players=args.players, map_type=args.map, vps_to_win=args.vps)

    env = make_env(seed=0, **config)
    td = env.reset()
    print(f"config: {env.core.config}")
    print(f"observation {tuple(td[OBS].shape)}, action mask {tuple(td[ACTION_MASK].shape)}, "
          f"acting now: player_{int(td[ACTIVE].nonzero())}")
    for w, rate in _benchmark(args.workers, args.frames, **config).items():
        print(f"{w} process{'es' if w > 1 else ''}: {rate:.0f} steps/s (random legal moves, no network)")
