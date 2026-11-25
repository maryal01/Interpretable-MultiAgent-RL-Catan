"""Policy distillation via VIPER-style decision tree extraction.

Loads a trained policy checkpoint, rolls out games to collect (observation,
action) pairs, fits a shallow decision tree to each agent's decisions, and
prints the top splits alongside the greedy heuristic's logic.

Usage:
    python distill.py --checkpoint checkpoints/iter_0490.pt
    python distill.py --checkpoint checkpoints/iter_0490.pt --agent 0 --depth 4 --games 200
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, set_composite_lp_aggregate
from torchrl.envs.utils import step_mdp
from torchrl.modules import MaskedCategorical, MultiAgentMLP, ProbabilisticActor
from sklearn.tree import DecisionTreeClassifier, export_text

from catan_env import ACTION_MASK, ACTIVE, OBS, CatanConfig, CatanCore, make_env

warnings.filterwarnings("ignore")

N_AGENTS = 4
DEPTH    = 3
N_GAMES  = 100


# ------------------------------------------------------------------ infer dims
def infer_dims(checkpoint_path: str) -> tuple[int, int]:
    """Read obs_dim and n_actions from checkpoint weight shapes instead of hardcoding."""
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    weights = [v for k, v in ckpt["policy"].items() if "weight" in k]
    obs_dim   = int(weights[0].shape[-1])   # first layer:  [n_agents, hidden, obs_dim]
    n_actions = int(weights[-1].shape[-2])  # last layer:   [n_agents, n_actions, hidden]
    return obs_dim, n_actions


# ------------------------------------------------------------------ config
CONFIG = CatanConfig(num_players=N_AGENTS, map_type="BASE", vps_to_win=8)


def config_for(obs_dim: int, n_actions: int) -> CatanConfig:
    """Pick trading on/off so the env's shapes match the checkpoint."""
    for trading in (False, True):
        cfg = CatanConfig(num_players=N_AGENTS, map_type="BASE", vps_to_win=8, trading=trading)
        core = CatanCore(cfg)
        if len(core.feature_names) == obs_dim and len(core.action_table) == n_actions:
            return cfg
    raise ValueError(f"no CatanConfig matches obs_dim={obs_dim}, n_actions={n_actions}")


# ------------------------------------------------------------------ load policy
def load_policy(checkpoint_path: str) -> tuple:
    """Always loads onto CPU — MPS doesn't support the vmap ops MultiAgentMLP uses."""
    set_composite_lp_aggregate(False).set()

    obs_dim, n_actions = infer_dims(checkpoint_path)
    print(f"Inferred obs_dim: {obs_dim}, n_actions: {n_actions}")

    global CONFIG
    CONFIG = config_for(obs_dim, n_actions)
    print(f"Env config: trading={CONFIG.trading}")
    env = make_env(seed=0, **vars(CONFIG))

    policy_net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
        n_agent_outputs=n_actions,
        n_agents=N_AGENTS,
        centralised=False,
        share_params=False,
        device=torch.device("cpu"),
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    policy_net.load_state_dict(ckpt["policy"])
    policy_net.eval()

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "logits")],
    )
    policy = ProbabilisticActor(
        module=policy_module,
        in_keys={"logits": ("agents", "logits"), "mask": ("agents", "action_mask")},
        out_keys=[("agents", "action")],
        distribution_class=MaskedCategorical,
        return_log_prob=False,
    )
    return policy, env, obs_dim, n_actions


# ------------------------------------------------------------------ greedy action
def greedy_action_for_agent(policy_module, obs: torch.Tensor,
                             mask: torch.Tensor, agent_idx: int,
                             n_actions: int) -> int:
    """
    Run the policy_module (TensorDictModule wrapping MultiAgentMLP) on a single
    observation and return the greedy (argmax) action for agent_idx.

    obs:  [obs_dim]   (1-D, for one agent)
    mask: [n_actions] (1-D, bool)
    """
    # MultiAgentMLP expects [batch, n_agents, obs_dim]
    # We give it a batch of 1 with the same obs broadcast across all agent slots
    obs_batch = obs.unsqueeze(0).unsqueeze(0).expand(1, N_AGENTS, -1)  # [1, N, D]

    td = TensorDict(
        {"agents": {"observation": obs_batch}},
        batch_size=[1],
    )

    with torch.no_grad():
        td = policy_module(td)

    logits = td["agents", "logits"][0, agent_idx].clone()  # [n_actions]
    logits[~mask] = -1e9
    return int(logits.argmax().item())


# ------------------------------------------------------------------ rollout
def collect_decisions(
    policy, policy_module, env, obs_dim: int, n_actions: int,
    agent_idx: int, n_games: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Roll out n_games full games. At every step where agent_idx is active,
    record (observation, greedy_action) as the distillation dataset.
    The rest of the game is driven by the stochastic policy as normal.
    """
    observations, actions = [], []

    for seed in range(n_games):
        env._rng = np.random.default_rng(seed)
        td = env.reset()
        # env is on CPU already; ensure tensors are CPU
        td = td.cpu()

        while True:
            active = td[ACTIVE]
            # ACTIVE can be shape [N] or [N,1]; handle both
            active_flat = active.squeeze()
            active_agent = int(active_flat.nonzero(as_tuple=False).squeeze())

            if active_agent == agent_idx:
                obs  = td[OBS][agent_idx].cpu()          # [obs_dim]
                mask = td[ACTION_MASK][agent_idx].cpu()  # [n_actions]

                greedy = greedy_action_for_agent(
                    policy_module, obs, mask, agent_idx, n_actions
                )
                observations.append(obs.numpy())
                actions.append(greedy)

            # Drive the game forward with the stochastic policy
            with torch.no_grad():
                td = policy(td.cpu())
            td = env.step(td)

            if td["next", "done"].item():
                break
            td = step_mdp(td)

    return np.stack(observations), np.array(actions)


# ------------------------------------------------------------------ feature names
def get_feature_names(obs_dim: int) -> list[str]:
    core = CatanCore(CONFIG)
    names = list(core.feature_names)
    # Pad if the checkpoint's obs is larger than the feature list we know about
    if len(names) < obs_dim:
        names += [f"feature_{i}" for i in range(len(names), obs_dim)]
    return names[:obs_dim]


# ------------------------------------------------------------------ distill
def distill(
    observations: np.ndarray,
    actions: np.ndarray,
    feature_names: list[str],
    max_depth: int,
    agent_idx: int,
) -> DecisionTreeClassifier:
    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=10,
        random_state=0,
    )
    tree.fit(observations, actions)
    fidelity = (tree.predict(observations) == actions).mean()
    print(f"\nAgent {agent_idx} — tree fidelity: {fidelity:.1%}  "
          f"({len(observations)} decisions, depth {max_depth})")
    return tree


# ------------------------------------------------------------------ analysis
def print_analysis(
    tree: DecisionTreeClassifier,
    feature_names: list[str],
    actions: np.ndarray,
    agent_idx: int,
):
    print(f"\n{'='*60}")
    print(f"  Agent {agent_idx} — Decision Tree")
    print(f"{'='*60}")
    print(export_text(tree, feature_names=feature_names, max_depth=3, spacing=3))

    # Top splits by feature importance
    importances = tree.feature_importances_
    top_k = 8
    top_idx = np.argsort(importances)[::-1][:top_k]
    print(f"Top {top_k} features by split importance:")
    for rank, i in enumerate(top_idx, 1):
        if importances[i] > 0:
            print(f"  {rank}. {feature_names[i]:<55s}  {importances[i]:.4f}")

    # Most common greedy actions
    core = CatanCore(CONFIG)
    print(f"\nTop 5 most frequent greedy actions:")
    for action_idx, count in Counter(actions).most_common(5):
        if action_idx < len(core.action_table):
            action_type, value = core.action_table[action_idx]
            name = f"{action_type.name}({value})"
        else:
            name = f"action_{action_idx}"
        print(f"  {name:<55s}  {count} ({count/len(actions):.1%})")

    # Heuristic vs leader-aware comparison
    top_features = [feature_names[i] for i in top_idx if importances[i] > 0]
    heuristic_kw = ["WOOD", "BRICK", "ORE", "GRAIN", "SHEEP",
                    "SETTLEMENT", "CITY", "ROAD", "VP", "VICTORY"]
    leader_kw    = ["LEADER", "PHASE", "TURN", "LARGEST", "LONGEST"]

    heuristic_hits = [f for f in top_features if any(k in f.upper() for k in heuristic_kw)]
    leader_hits    = [f for f in top_features if any(k in f.upper() for k in leader_kw)]

    print(f"\nHeuristic comparison:")
    if heuristic_hits:
        print(f"  Splits on resource/build features: {heuristic_hits[:3]}")
        print("  -> Agent may have rediscovered greedy build-order heuristic.")
    if leader_hits:
        print(f"  Splits on game-state features: {leader_hits[:3]}")
        print("  -> Agent may have learned leader-aware or phase-aware strategy.")
    if not heuristic_hits and not leader_hits:
        print("  -> No clear match to known heuristics; inspect the tree above.")


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--agent", type=int, default=None,
                        help="Agent index 0-3. Default: all agents.")
    parser.add_argument("--depth", type=int, default=DEPTH,
                        help=f"Max tree depth (default {DEPTH}).")
    parser.add_argument("--games", type=int, default=N_GAMES,
                        help=f"Rollout games (default {N_GAMES}).")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    policy, env, obs_dim, n_actions = load_policy(args.checkpoint)

    #policy, env, obs_dim, n_actions = load_policy(args.checkpoint)
    print(type(policy))
    print(type(policy.module))
    for name, mod in policy.named_children():
        print(name, type(mod))

    # policy_module is the TensorDictModule wrapping MultiAgentMLP
    policy_module = policy.module[0]

    feature_names = get_feature_names(obs_dim)
    agents_to_distill = [args.agent] if args.agent is not None else list(range(N_AGENTS))

    for agent_idx in agents_to_distill:
        print(f"\nCollecting decisions for agent {agent_idx} over {args.games} games...")
        observations, actions = collect_decisions(
            policy, policy_module, env, obs_dim, n_actions, agent_idx, args.games
        )
        print(f"  Collected {len(observations)} decision points.")

        if len(observations) < 20:
            print(f"  Too few samples for agent {agent_idx} — skipping.")
            continue

        tree = distill(observations, actions, feature_names, args.depth, agent_idx)
        print_analysis(tree, feature_names, actions, agent_idx)


if __name__ == "__main__":
    main()