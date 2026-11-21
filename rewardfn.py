# import numpy as np
# from collections import defaultdict
# from catanatron.models.decks import CITY_COST_FREQDECK, ROAD_COST_FREQDECK, SETTLEMENT_COST_FREQDECK
# from catanatron.models.enums import SETTLEMENT, ActionType
# from catanatron.state_functions import get_actual_victory_points, get_player_buildings, get_player_freqdeck, player_key

# BUILDS = {
#     "settlement": (np.array(SETTLEMENT_COST_FREQDECK), 1.0),
#     "city": (np.array(CITY_COST_FREQDECK), 1.0),
#     "road": (np.array(ROAD_COST_FREQDECK), 0.25)
# }

# def build_limits(game, color):
#     key = player_key(game.state, color)
#     pieces = lambda piece: game.state.player_state[f"{key}_{piece}_AVAILABLE"]
#     return {
#         "settlement": min(pieces("SETTLEMENTS"), len(game.state.board.buildable_node_ids(color))),
#         "city": min(pieces("CITIES"), len(get_player_buildings(game.state, color, SETTLEMENT))),
#         "road": min(pieces("ROADS"), len(game.state.board.buildable_edges(color))),
#     }

# def hand_utility(hand, limits, seven_penalty=0.0):
#     left, utility = np.array(hand), 0.0
#     for build, (cost, value) in BUILDS.items():
#         count = min(limits[build], min(left[cost > 0] // cost[cost > 0]))
#         left = left - count * cost
#         utility += count * value
    
#     partial = 0.0
#     for build, (cost, value) in BUILDS.items():
#         if limits[build]:
#             covered = np.minimum(left, cost).sum() / cost.sum()
#             partial = max(partial, value * covered)
#     utility += partial

#     if sum(hand) > 7:
#         utility -= seven_penalty * (sum(hand) // 2) / 6
#     return float(utility)

# def trade_gains(game, action, seven_penalty=0.0):
#     gives, asks, partner = np.array(action.value[:5]), np.array(action.value[5:10]), action.value[10]
#     gains = {}
#     for color, received in ((action.color, asks - gives), (partner, gives - asks)):
#         after = np.array(get_player_freqdeck(game.state, color))
#         limits = build_limits(game, color)
#         after_trade = hand_utility(after, limits, seven_penalty)
#         before_trade = hand_utility(after - received, limits, seven_penalty)

#         gains[color] = after_trade - before_trade
#     return gains


# class InitialRewardFunction:
#     def __init__(self, vp_weight = 0.1, win = 1, loss = -1, trade_weight = 0.0, seven_penalty = 0.0):
#         self.vp_weight = vp_weight
#         self.loss_reward = loss
#         self.win_reward = win
#         self.trade_weight = trade_weight
#         self.seven_penalty = seven_penalty

#     def reset(self, core, game):
#         self.prev_vp = defaultdict(float)

#     def __call__(self, core, game):
#         rewards = np.zeros(core.n_agents, dtype=np.float32)

#         for i, color in enumerate(core.colors):
#             current = get_actual_victory_points(game.state, color)
#             rewards[i] = self.vp_weight * (current - self.prev_vp[color])
#             self.prev_vp[color] = current

#         if self.trade_weight and game.state.action_records:
#             action = game.state.action_records[-1].action
#             if action.action_type == ActionType.CONFIRM_TRADE:
#                 for color, gain in trade_gains(game, action, self.seven_penalty).items():
#                     rewards[core.colors.index(color)] += self.trade_weight * gain

#         winner = game.winning_color()
#         if winner is not None:
#             rewards[:] += self.loss_reward
#             winning_index = core.colors.index(winner)
#             rewards[winning_index] += self.win_reward - self.loss_reward

#         return rewards


import numpy as np
from collections import defaultdict
from catanatron.models.decks import CITY_COST_FREQDECK, ROAD_COST_FREQDECK, SETTLEMENT_COST_FREQDECK
from catanatron.models.enums import CITY, SETTLEMENT, ActionType
from catanatron.state_functions import (get_actual_victory_points, get_largest_army,
                                        get_longest_road_color, get_player_buildings,
                                        get_player_freqdeck, player_key)

BUILDS = {
    "settlement": (np.array(SETTLEMENT_COST_FREQDECK), 1.0),
    "city": (np.array(CITY_COST_FREQDECK), 1.0),
    "road": (np.array(ROAD_COST_FREQDECK), 0.25)
}

def build_limits(game, color):
    key = player_key(game.state, color)
    pieces = lambda piece: game.state.player_state[f"{key}_{piece}_AVAILABLE"]
    return {
        "settlement": min(pieces("SETTLEMENTS"), len(game.state.board.buildable_node_ids(color))),
        "city": min(pieces("CITIES"), len(get_player_buildings(game.state, color, SETTLEMENT))),
        "road": min(pieces("ROADS"), len(game.state.board.buildable_edges(color))),
    }

def hand_utility(hand, limits, seven_penalty=0.0):
    left, utility = np.array(hand), 0.0
    for build, (cost, value) in BUILDS.items():
        count = min(limits[build], min(left[cost > 0] // cost[cost > 0]))
        left = left - count * cost
        utility += count * value

    partial = 0.0
    for build, (cost, value) in BUILDS.items():
        if limits[build]:
            covered = np.minimum(left, cost).sum() / cost.sum()
            partial = max(partial, value * covered)
    utility += partial

    if sum(hand) > 7:
        utility -= seven_penalty * (sum(hand) // 2) / 6
    return float(utility)

def trade_gains(game, action, seven_penalty=0.0):
    gives, asks, partner = np.array(action.value[:5]), np.array(action.value[5:10]), action.value[10]
    gains = {}
    for color, received in ((action.color, asks - gives), (partner, gives - asks)):
        after = np.array(get_player_freqdeck(game.state, color))
        limits = build_limits(game, color)
        after_trade = hand_utility(after, limits, seven_penalty)
        before_trade = hand_utility(after - received, limits, seven_penalty)

        gains[color] = after_trade - before_trade
    return gains


class InitialRewardFunction:
    def __init__(self, vp_weight = 0.1, win = 1, loss = -1, trade_weight = 0.0, seven_penalty = 0.0):
        self.vp_weight = vp_weight
        self.loss_reward = loss
        self.win_reward = win
        self.trade_weight = trade_weight
        self.seven_penalty = seven_penalty

    def reset(self, core, game):
        self.prev_vp = defaultdict(float)

    def __call__(self, core, game):
        rewards = np.zeros(core.n_agents, dtype=np.float32)

        for i, color in enumerate(core.colors):
            current = get_actual_victory_points(game.state, color)
            rewards[i] = self.vp_weight * (current - self.prev_vp[color])
            self.prev_vp[color] = current

        if self.trade_weight and game.state.action_records:
            action = game.state.action_records[-1].action
            if action.action_type == ActionType.CONFIRM_TRADE:
                for color, gain in trade_gains(game, action, self.seven_penalty).items():
                    rewards[core.colors.index(color)] += self.trade_weight * gain

        winner = game.winning_color()
        if winner is not None:
            rewards[:] += self.loss_reward
            winning_index = core.colors.index(winner)
            rewards[winning_index] += self.win_reward - self.loss_reward

        return rewards


class SimpleCatanReward:
    def __init__(self, vp_weight=0.1, win=1.0, loss=-1.0, truncation=-1.2):
        self.vp_weight = vp_weight
        self.loss_reward = loss
        self.win_reward = win
        self.truncation_reward = truncation     # paid when the turn limit is hit

    def points(self, game, color):              # VP counted rule by rule, no VP cards
        state = game.state
        pts = len(get_player_buildings(state, color, SETTLEMENT))   # 1 VP each
        pts += 2 * len(get_player_buildings(state, color, CITY))    # 2 VP each
        if get_longest_road_color(state) == color:                  # 2 VP
            pts += 2
        if get_largest_army(state)[0] == color:                     # 2 VP
            pts += 2
        return pts

    def reset(self, core, game):
        self.prev_vp = {c: self.points(game, c) for c in core.colors}
        self.finished = False                   # end-of-game reward paid once

    def __call__(self, core, game):
        rewards = np.zeros(core.n_agents, dtype=np.float32)

        for i, color in enumerate(core.colors):
            current = self.points(game, color)
            rewards[i] = self.vp_weight * (current - self.prev_vp[color])
            self.prev_vp[color] = current

        terminated, truncated = core.status(game)
        if (terminated or truncated) and not self.finished:
            self.finished = True
            winner = game.winning_color()
            if winner is None:
                rewards[:] += self.truncation_reward
            else:
                rewards[:] += self.loss_reward
                winning_index = core.colors.index(winner)
                rewards[winning_index] += self.win_reward - self.loss_reward

        return rewards


REWARD_FUNCTIONS = {
    "initial": InitialRewardFunction,
    "simple": SimpleCatanReward,
}

def make_reward_fn(name="initial", **kwargs):
    """Build a reward function by name. kwargs go to that class's __init__."""
    if name not in REWARD_FUNCTIONS:
        raise ValueError(f"unknown reward function {name!r}; choose from {sorted(REWARD_FUNCTIONS)}")
    return REWARD_FUNCTIONS[name](**kwargs)