import functools

import numpy as np
import pytest
import torch
from torchrl.collectors import Collector
from torchrl.envs import check_env_specs
from torchrl.envs.utils import step_mdp

from catanatron.models.enums import ActionType
from catan_env import (ACTION, ACTION_MASK, ACTIVE, OBS, REWARD, CatanConfig, CatanCore, IllegalActionError,
                       _masked_random_policy, make_env)
from evaluate import play_match, random_policy
from rewardfn import InitialRewardFunction

CONFIGS = [CatanConfig(num_players=3), CatanConfig(num_players=4), CatanConfig(num_players=4, map_type="BASE",
                                                                                vps_to_win=10, turn_limit=1000)]
IDS = ["3p-mini", "4p-mini", "4p-base"]


def play_random_game(core, seed, check=lambda game: None):
    rng = np.random.default_rng(seed)
    game = core.new_game(seed)
    while not core.is_over(game):
        check(game)
        core.apply(game, core.current_agent(game), int(rng.choice(core.legal_indices(game))))
    return game

# -------------------------------------------------------------------- core
@pytest.mark.parametrize("dev_cards", [True, False])
def test_dev_cards_flag(dev_cards):
    core = CatanCore(CatanConfig(dev_cards=dev_cards))
    buy = core.action_table.index((ActionType.BUY_DEVELOPMENT_CARD, None))
    offered = []  # was "buy a dev card" a legal move at each step?

    def check(game):
        offered.append(core.legal_action_mask(game, core.current_agent(game))[buy])

    bought = 0
    for seed in range(3):
        game = play_random_game(core, seed, check)
        bought += sum(r.action.action_type == ActionType.BUY_DEVELOPMENT_CARD for r in game.state.action_records)

    if dev_cards:
        assert bought > 0, "control: random play does buy cards when they're on"
    else:
        assert bought == 0 and not any(offered)
        
@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_masks_and_action_indices(config):
    core = CatanCore(config)
    robber_steals = 0

    def check(game):
        nonlocal robber_steals
        agent, order = core.current_agent(game), game.state.colors
        mask = core.legal_action_mask(game, agent)
        assert set(np.flatnonzero(mask)) == {core.action_to_index(game, a) for a in game.playable_actions}
        for a in game.playable_actions:
            assert core.index_to_action(game, core.action_to_index(game, a), agent) == a  # round trip
            if a.action_type == ActionType.MOVE_ROBBER and a.value[1] is not None:  # relative victims
                offset = (order.index(a.value[1]) - order.index(a.color)) % core.n_agents
                assert core.action_table[core.action_to_index(game, a)][1][1] == core.colors[offset]
                robber_steals += 1
        assert not any(core.legal_action_mask(game, o).any() for o in range(core.n_agents) if o != agent)

    for seed in range(2):
        play_random_game(core, seed, check)
    assert robber_steals > 0


def test_same_seed_same_game():
    core = CatanCore()

    def trace(seed):
        obs = []
        play_random_game(core, seed, lambda g: obs.append(core.observation(g, core.current_agent(g))))
        return np.stack(obs)

    assert np.array_equal(trace(7), trace(7))


def test_illegal_actions_raise():
    core = CatanCore()
    game = core.new_game(0)
    agent = core.current_agent(game)
    legal = set(core.legal_indices(game))
    with pytest.raises(IllegalActionError):
        core.apply(game, (agent + 1) % core.n_agents, next(iter(legal)))  # not your turn
    with pytest.raises(IllegalActionError):
        core.apply(game, agent, next(i for i in range(core.n_actions) if i not in legal))


# --------------------------------------------------------------------- env
@pytest.mark.parametrize("config", CONFIGS, ids=IDS)
def test_specs_and_full_game(config):
    env = make_env(seed=1, **vars(config))
    check_env_specs(env)
    td = env.rollout(5000, policy=_masked_random_policy(), break_when_any_done=True)
    n = config.num_players
    assert td[OBS].shape[-2] == n and (td[ACTIVE].sum(-1) == 1).all()
    chosen = td[ACTION_MASK].gather(-1, td[ACTION].unsqueeze(-1)).squeeze(-1)
    assert chosen[td[ACTIVE]].all(), "the acting player's action is always legal"
    assert td["next", "done"][-1].all() and not td["next", "done"][:-1].any()
    assert not td["next", *REWARD][:-1].any()
    if td["next", "terminated"][-1].all():
        assert sorted(td["next", *REWARD][-1].squeeze(-1).tolist()) == [-1.0] * (n - 1) + [1.0]


def test_env_matches_core_step_by_step():
    """The env reports exactly what the core says for a game with the same seed and moves."""
    env = make_env(seed=123)
    td = env.reset()
    core = CatanCore()
    game = core.new_game(env.last_game_seed)
    rng = np.random.default_rng(0)
    while True:
        agent = core.current_agent(game)
        assert bool(td[ACTIVE][agent]) and int(td[ACTIVE].sum()) == 1
        for i in range(core.n_agents):
            assert np.array_equal(td[OBS][i].numpy(), core.observation(game, i))
        assert np.array_equal(td[ACTION_MASK][agent].numpy(), core.legal_action_mask(game, agent))
        action = int(rng.choice(core.legal_indices(game)))
        core.apply(game, agent, action)
        td = env.step(td.set(ACTION, torch.full((core.n_agents,), action)))
        assert np.array_equal(td["next", *REWARD].squeeze(-1).numpy(), core.rewards(game))
        assert bool(td["next", "done"]) == core.is_over(game)
        if core.is_over(game):
            break
        td = step_mdp(td)


def test_illegal_action_in_env_raises():
    env = make_env(seed=0)
    td = env.reset()
    agent = int(td[ACTIVE].nonzero())
    illegal = int((~td[ACTION_MASK][agent]).nonzero()[0])
    with pytest.raises(IllegalActionError):
        env.step(td.set(ACTION, torch.full((4,), illegal)))


def test_auto_reset_and_multiprocess_collector():
    env = make_env(seed=2)
    td = env.rollout(2500, policy=_masked_random_policy(), break_when_any_done=False)
    assert td["next", "done"].sum() >= 2 and (td[ACTIVE].sum(-1) == 1).all()

    envs = [functools.partial(make_env, seed=s) for s in range(2)]
    collector = Collector(envs, _masked_random_policy(), num_collectors=2, sync=True,
                          frames_per_batch=400, total_frames=800)
    frames = sum(batch.numel() for batch in collector)
    collector.shutdown()
    assert frames >= 800


# -------------------------------------------------------------------- eval
@pytest.mark.parametrize("n", [3, 4])
def test_play_match(n):
    result = play_match([random_policy(0)] + ["random"] * (n - 1), n_games=4, config=CatanConfig(num_players=n))
    assert result.wins.sum() + result.truncated == 4 and len(result.robber) == n
    for r in result.robber:
        assert r["steals_from_leader"] <= r["clear_leader_steals"] <= r["steals"]


def test_play_match_with_trading():
    seats = [random_policy(0), random_policy(1), "random", "random"]  # policies offer, policies and bots answer
    result = play_match(seats, n_games=2, config=CatanConfig(trading=True))
    assert result.wins.sum() + result.truncated == 2
    assert sum(t["offers"] for t in result.trade) > 0 and "yes if good" in result.table()
    for t in result.trade:
        assert t["helped"] + t["hurt"] <= t["trades"] and t["built_after"] <= t["trades"]
        assert t["yes_good"] <= t["asked_good"] and t["asked_good"] + t["asked_bad"] <= t["accepts"] + t["rejects"]


def test_illegal_policy_is_caught():
    with pytest.raises(IllegalActionError):
        play_match([lambda obs, mask: int((~mask).nonzero()[0][0])] + ["random"] * 3, n_games=1)


# ------------------------------------------------------------------ reward
def test_trade_reward_only_pays_the_two_traders():
    reward_fn = InitialRewardFunction(vp_weight=0, win=0, loss=0, trade_weight=1.0)  # trade part only
    core = CatanCore(CatanConfig(trading=True), reward_fn=reward_fn)
    rng, game, trades = np.random.default_rng(0), core.new_game(0), 0
    while not core.is_over(game):
        action = core.apply(game, core.current_agent(game), int(rng.choice(core.legal_indices(game))))
        rewards = core.rewards(game)
        if action.action_type == ActionType.CONFIRM_TRADE:
            traders = [core.colors.index(c) for c in (action.color, action.value[10])]
            others = [i for i in range(core.n_agents) if i not in traders]
            assert not rewards[others].any()
            trades += 1
        else:
            assert not rewards.any()
    assert trades > 0
