"""Profile what kind of Catan strategy each seat plays.

Plays full games (by default the checkpoint's 4 networks against each other, i.e.
self-play) and summarizes each seat's behavior along the axes that separate Catan
strategies: opening placement, build mix, tempo, bank trades, discards, passivity,
robber use, player trades, and where its victory points came from.

    python profile_policies.py checkpoints_simple_notrade/iter_0199.pt --games 100
    python profile_policies.py checkpoints_simple/iter_0199.pt --seats this value value value
    python profile_policies.py --seats value value value value          # reference: the value bot
    python profile_policies.py CKPT --json distill_results/profile.json  # also save numbers

--seats takes one entry per seat: "this" (seat k is played by the checkpoint's network k),
a bot name (random, weighted, victory_point, value, alphabeta) or another checkpoint path.
Trading on/off is read from the checkpoint's weight shapes; no flag is needed.

Policies sample from their action distribution, as during training. Pass --greedy to
always take the most likely action instead (what runeval.py does).

Units: a "round" is one turn for every player (Catanatron counts single turns).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from collections import Counter, defaultdict

import numpy as np
import torch
from catanatron.models.enums import CITY, SETTLEMENT, ActionType
from catanatron.observer import GameObserver
from catanatron.state_functions import (get_actual_victory_points, get_largest_army,
                                        get_longest_road_color, get_player_buildings)

from catan_env import CatanConfig, CatanCore, RobberStats, TradeStats
from evaluate import BOTS, PolicyPlayer, make_policy
from runeval import load_policy_net

BASE = CatanConfig(num_players=4, map_type="BASE", vps_to_win=8)
RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
BUILD_TYPES = {ActionType.BUILD_ROAD: "road", ActionType.BUILD_SETTLEMENT: "settlement",
               ActionType.BUILD_CITY: "city", ActionType.BUY_DEVELOPMENT_CARD: "dev_card"}
VP_BUILDS = {ActionType.BUILD_SETTLEMENT, ActionType.BUILD_CITY}


def pips(number) -> int:
    """Dots on a number token: how many of the 36 dice outcomes produce it."""
    return 0 if number is None else 6 - abs(7 - number)


def config_for_checkpoint(path: str) -> CatanConfig:
    """Pick trading on/off so the game's shapes match the checkpoint's networks."""
    weights = [v for k, v in torch.load(path, map_location="cpu")["policy"].items() if "weight" in k]
    obs_dim, n_actions = int(weights[0].shape[-1]), int(weights[-1].shape[-2])
    for trading in (False, True):
        cfg = dataclasses.replace(BASE, trading=trading)
        core = CatanCore(cfg)
        if (core.obs_dim, core.n_actions) == (obs_dim, n_actions):
            return cfg
    raise SystemExit(f"{path}: no config matches obs_dim={obs_dim}, n_actions={n_actions}")


# ------------------------------------------------------------------ observer
class ProfileStats(GameObserver):
    """Per-color behavior counts for one game. step() sees the game before each action."""

    def before(self, game):
        self.n = len(game.state.colors)
        self.c = defaultdict(Counter)            # color -> counters
        self.opening_pips = defaultdict(Counter)  # color -> resource -> pips of opening settlements
        self.settlements = Counter()              # settlements built so far, incl. opening ones
        self.first = defaultdict(dict)            # color -> event -> round it happened

    def _round(self, game) -> float:
        return game.state.num_turns / self.n

    def step(self, game, action):
        kind, color, state = action.action_type, action.color, game.state
        c = self.c[color]
        choices = len(game.playable_actions)
        c["decisions"] += 1
        if choices > 1:  # trade offers aren't listed by the engine, so this slightly undercounts
            c["real_choices"] += 1
            c[f"choice:{kind.name}"] += 1

        if state.is_initial_build_phase:
            if kind == ActionType.BUILD_SETTLEMENT:
                self.settlements[color] += 1
                for tile in state.board.map.adjacent_tiles[action.value]:
                    if tile.resource is not None:
                        self.opening_pips[color][tile.resource] += pips(tile.number)
            return  # opening roads/settlements are free: not part of the build mix

        if kind in BUILD_TYPES:
            c[f"build:{BUILD_TYPES[kind]}"] += 1
            if kind == ActionType.BUILD_SETTLEMENT:
                self.settlements[color] += 1
                if self.settlements[color] == 3:
                    self.first[color].setdefault("third_settlement", self._round(game))
            elif kind == ActionType.BUILD_CITY:
                self.first[color].setdefault("first_city", self._round(game))
        elif kind == ActionType.MARITIME_TRADE:
            give = [r for r in action.value[:4] if r is not None]
            c[f"bank_{len(give)}to1"] += 1
            c["bank_trades"] += 1
            c[f"bank_get:{action.value[4]}"] += 1
            c[f"bank_give:{give[0]}"] += 1
        elif kind == ActionType.DISCARD_RESOURCE:
            c["cards_discarded"] += 1
        elif kind == ActionType.END_TURN:
            c["end_turns"] += 1
            can = {a.action_type for a in game.playable_actions}
            c["end_turn_could_build_vp"] += bool(can & VP_BUILDS)
            c["end_turn_could_build_any"] += bool(can & set(BUILD_TYPES))

    def finish(self, game) -> dict:
        """Per-color record for this game, including where final VPs came from."""
        state, winner = game.state, game.winning_color()
        lr, la = get_longest_road_color(state), get_largest_army(state)[0]
        out = {}
        for color in state.colors:
            s = len(get_player_buildings(state, color, SETTLEMENT))
            ct = len(get_player_buildings(state, color, CITY))
            total = get_actual_victory_points(state, color)
            vp = {"settlements": s, "cities": 2 * ct, "longest_road": 2 * (lr == color),
                  "largest_army": 2 * (la == color)}
            vp["vp_cards"] = total - sum(vp.values())
            out[color] = {
                "won": winner == color,
                "final_vp": total,
                "vp_from": vp,
                "counts": dict(self.c[color]),
                "opening_pips": dict(self.opening_pips[color]),
                "first": dict(self.first[color]),
                "win_round": self._round(game) if winner == color else None,
            }
        return out


# ------------------------------------------------------------------ playing
def play(seats, n_games, config, seed):
    core = CatanCore(config)
    records = [[] for _ in range(core.n_agents)]
    truncated, rounds = 0, []
    for g in range(n_games):
        players = [BOTS[s](c) if isinstance(s, str) else PolicyPlayer(c, core, s)
                   for s, c in zip(seats, core.colors)]
        game = core.new_game(seed + g, players=players)
        prof, robber, trade = ProfileStats(), RobberStats(), TradeStats()
        for obs in (prof, robber, trade):
            game.watch(obs)
        while not core.is_over(game):
            game.play_tick()
            core.skip_forced_rejects(game)
        truncated += game.winning_color() is None
        rounds.append(game.state.num_turns / core.n_agents)
        per_color = prof.finish(game)
        rob, trd = robber.summary(), trade.summary()
        for i, color in enumerate(core.colors):
            rec = per_color[color]
            rec["robber"] = rob.get(color.value, dict.fromkeys(RobberStats.FIELDS, 0))
            rec["trade"] = trd.get(color.value, dict.fromkeys(TradeStats.FIELDS, 0))
            records[i].append(rec)
        print(f"\r  game {g + 1}/{n_games}", end="", flush=True)
    print()
    return records, truncated, rounds


# ------------------------------------------------------------------ summarizing
def summarize(recs: list[dict]) -> dict:
    """Collapse one seat's per-game records into the profile numbers."""
    n = len(recs)
    tot = Counter()
    for r in recs:
        tot.update(r["counts"])
    share = lambda a, b: a / b if b else float("nan")
    mean_of = lambda key: (lambda xs: float(np.mean(xs)) if xs else float("nan"))(
        [r["first"][key] for r in recs if key in r["first"]])

    opening = Counter()
    for r in recs:
        opening.update(r["opening_pips"])
    opening_total = sum(opening.values())
    builds = {b: tot[f"build:{b}"] for b in ("road", "settlement", "city", "dev_card")}
    vp_from = Counter()
    for r in recs:
        vp_from.update(r["vp_from"])
    robber, trade = Counter(), Counter()
    for r in recs:
        robber.update(r["robber"])
        trade.update(r["trade"])
    wins = [r for r in recs if r["won"]]
    real = tot["real_choices"]

    return {
        "games": n,
        "win_rate": len(wins) / n,
        "final_vp": float(np.mean([r["final_vp"] for r in recs])),
        "win_round": float(np.mean([r["win_round"] for r in wins])) if wins else float("nan"),
        "opening_pip_share": {res: share(opening[res], opening_total) for res in RESOURCES},
        "opening_pips_per_game": opening_total / n,
        "builds_per_game": {b: v / n for b, v in builds.items()},
        "build_share": {b: share(v, sum(builds.values())) for b, v in builds.items()},
        "city_share_of_vp_builds": share(builds["city"], builds["city"] + builds["settlement"]),
        "third_settlement_round": mean_of("third_settlement"),
        "reached_third_settlement": sum("third_settlement" in r["first"] for r in recs) / n,
        "first_city_round": mean_of("first_city"),
        "reached_first_city": sum("first_city" in r["first"] for r in recs) / n,
        "bank_trades_per_game": tot["bank_trades"] / n,
        "bank_ratio_share": {k: share(tot[f"bank_{k}to1"], tot["bank_trades"]) for k in (4, 3, 2)},
        "bank_get": {res: tot[f"bank_get:{res}"] for res in RESOURCES},
        "bank_give": {res: tot[f"bank_give:{res}"] for res in RESOURCES},
        "cards_discarded_per_game": tot["cards_discarded"] / n,
        "end_turn_with_vp_build": share(tot["end_turn_could_build_vp"], tot["end_turns"]),
        "end_turn_with_any_build": share(tot["end_turn_could_build_any"], tot["end_turns"]),
        "real_choice_share": share(real, tot["decisions"]),
        "real_choice_mix": {k.split(":", 1)[1]: v / real for k, v in tot.items()
                            if k.startswith("choice:")} if real else {},
        "vp_from_per_game": {k: v / n for k, v in vp_from.items()},
        "robber": dict(robber),
        "trade": dict(trade),
    }


def style(p: dict) -> list[str]:
    """Rough, rule-of-thumb labels. Thresholds are deliberately simple; check the numbers."""
    tags = []
    o = p["opening_pip_share"]
    city_res, road_res = o["ORE"] + o["WHEAT"], o["WOOD"] + o["BRICK"]
    if city_res >= 0.5:
        tags.append("ore/wheat opening")
    elif road_res >= 0.5:
        tags.append("wood/brick opening")
    cs = p["city_share_of_vp_builds"]
    if cs == cs:  # not NaN
        tags.append("city-focused" if cs >= 0.6 else "settlement-focused" if cs <= 0.4 else "balanced builds")
    if p["builds_per_game"]["road"] >= 2 * max(1e-9, p["builds_per_game"]["settlement"] + p["builds_per_game"]["city"]):
        tags.append("road-heavy")
    if p["end_turn_with_vp_build"] >= 0.3:
        tags.append("passive (often ends turn when it could build VP)")
    r = p["robber"]
    if r.get("clear_leader_steals", 0) >= 10:
        ratio = r["steals_from_leader"] / max(1e-9, r["chance_from_leader"])
        if ratio >= 1.3:
            tags.append("targets the leader with robber")
        elif ratio <= 0.7:
            tags.append("avoids robbing the leader")
    if p["trade"].get("offers", 0) / p["games"] >= 3:
        tags.append("active trader")
    return tags


# ------------------------------------------------------------------ printing
def fmt(x, pct=False, d=1):
    if x is None or x != x:
        return "-"
    return f"{x:.0%}" if pct else f"{x:.{d}f}"


def print_report(labels, profiles, truncated, rounds, n_games, config):
    w = 13
    head = lambda title: print(f"\n{title}\n{'':<30}" + "".join(f"{l:>{w}}" for l in labels))
    row = lambda name, vals: print(f"{name:<30}" + "".join(f"{v:>{w}}" for v in vals))

    print(f"\n{n_games} games, trading={config.trading}, {truncated} hit the turn limit, "
          f"mean {np.mean(rounds):.1f} rounds")

    head("OUTCOME")
    row("win rate", [fmt(p["win_rate"], pct=True) for p in profiles])
    row("final VP", [fmt(p["final_vp"]) for p in profiles])
    row("round of win (when won)", [fmt(p["win_round"]) for p in profiles])
    for src in ("settlements", "cities", "longest_road", "largest_army", "vp_cards"):
        row(f"  VP from {src}", [fmt(p["vp_from_per_game"].get(src, 0), d=2) for p in profiles])

    head("OPENING (share of production pips on the 2 opening settlements)")
    for res in RESOURCES:
        row(f"  {res.lower()}", [fmt(p["opening_pip_share"][res], pct=True) for p in profiles])
    row("total pips per game", [fmt(p["opening_pips_per_game"]) for p in profiles])

    head("BUILDING (after the opening; per game)")
    for b, name in (("road", "roads"), ("settlement", "settlements"), ("city", "cities"), ("dev_card", "dev cards")):
        row(f"  {name}", [fmt(p["builds_per_game"][b], d=2) for p in profiles])
    row("cities / (cities+settlements)", [fmt(p["city_share_of_vp_builds"], pct=True) for p in profiles])

    head("TEMPO (round reached, in games where it happened)")
    row("3rd settlement", [fmt(p["third_settlement_round"]) for p in profiles])
    row("  ...reached in", [fmt(p["reached_third_settlement"], pct=True) for p in profiles])
    row("1st city", [fmt(p["first_city_round"]) for p in profiles])
    row("  ...reached in", [fmt(p["reached_first_city"], pct=True) for p in profiles])

    head("RESOURCES")
    row("bank trades per game", [fmt(p["bank_trades_per_game"], d=2) for p in profiles])
    for k in (4, 3, 2):
        row(f"  at {k}:1", [fmt(p["bank_ratio_share"][k], pct=True) for p in profiles])
    row("most wanted from bank", [max(p["bank_get"], key=p["bank_get"].get).lower()
                                  if sum(p["bank_get"].values()) else "-" for p in profiles])
    row("cards discarded per game", [fmt(p["cards_discarded_per_game"], d=2) for p in profiles])

    head("DECISIONS")
    row("decisions with >1 option", [fmt(p["real_choice_share"], pct=True) for p in profiles])
    row("end turn when VP build legal", [fmt(p["end_turn_with_vp_build"], pct=True) for p in profiles])
    row("end turn when any build legal", [fmt(p["end_turn_with_any_build"], pct=True) for p in profiles])
    kinds = sorted({k for p in profiles for k in p["real_choice_mix"]},
                   key=lambda k: -max(p["real_choice_mix"].get(k, 0) for p in profiles))[:8]
    for k in kinds:
        row(f"  {k.lower()}", [fmt(p["real_choice_mix"].get(k, 0), pct=True) for p in profiles])

    head("ROBBER (steals while one opponent clearly led)")
    row("steals", [str(p["robber"].get("clear_leader_steals", 0)) for p in profiles])
    row("  from the leader", [str(p["robber"].get("steals_from_leader", 0)) for p in profiles])
    row("  expected by chance", [fmt(p["robber"].get("chance_from_leader", 0)) for p in profiles])

    if config.trading:
        head("PLAYER TRADES (totals)")
        for k in TradeStats.FIELDS:
            row(f"  {k}", [str(p["trade"].get(k, 0)) for p in profiles])

    print("\nSTYLE (rule-of-thumb labels; see style() for thresholds)")
    for label, p in zip(labels, profiles):
        print(f"  {label:<14} {', '.join(style(p)) or '-'}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", nargs="?", help="checkpoint used for 'this' seats")
    ap.add_argument("--seats", nargs="+", default=["this"] * 4)
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--greedy", action="store_true", help="argmax actions instead of sampling")
    ap.add_argument("--json", help="also write the profiles (and per-game records) to this file")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    paths = [args.checkpoint if s == "this" else s for s in args.seats if s not in BOTS]
    if any(p is None for p in paths):
        raise SystemExit("'this' seats need a checkpoint argument")
    configs = {config_for_checkpoint(p) for p in set(paths)}
    if len(configs) > 1:
        raise SystemExit("all checkpoints must share the same trading setting")
    config = configs.pop() if configs else BASE
    core = CatanCore(config)
    if len(args.seats) != core.n_agents:
        raise SystemExit(f"--seats needs {core.n_agents} entries")

    nets, seats = {}, []
    for k, who in enumerate(args.seats):
        if who in BOTS:
            seats.append(who)
            continue
        path = args.checkpoint if who == "this" else who
        if path not in nets:
            nets[path], it = load_policy_net(path, core)
            print(f"{path} (iteration {it})")
        seats.append(make_policy(nets[path], k, core.n_agents, deterministic=args.greedy))
    labels = [s if s in BOTS else f"agent{k}" if s == "this" else f"{pathlib.Path(s).stem[:7]}/{k}"
              for k, s in enumerate(args.seats)]

    records, truncated, rounds = play(seats, args.games, config, args.seed)
    profiles = [summarize(r) for r in records]
    print_report(labels, profiles, truncated, rounds, args.games, config)

    if args.json:
        out = pathlib.Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "checkpoint": args.checkpoint, "seats": args.seats, "games": args.games, "seed": args.seed,
            "greedy": args.greedy, "trading": config.trading, "truncated": truncated,
            "mean_rounds": float(np.mean(rounds)),
            "profiles": dict(zip(labels, profiles)),
            "per_game": {l: recs for l, recs in zip(labels, records)},
        }, indent=1, default=str))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
