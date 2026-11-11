import argparse
import dataclasses
import itertools
import torch
from torchrl.modules import MultiAgentMLP
from catan_env import CatanConfig, CatanCore
from evaluate import make_policy, play_match

CONFIG = CatanConfig(num_players=4, map_type="BASE", vps_to_win=8)

def load_policy_net(path, core):
    net = MultiAgentMLP(
        n_agent_inputs=core.obs_dim,
        n_agent_outputs=core.n_actions,
        n_agents=core.n_agents,
        centralised=False,
        share_params=False,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )
    ckpt = torch.load(path, map_location="cpu")
    try:
        net.load_state_dict(ckpt["policy"])
    except (RuntimeError, ValueError) as e:
        raise SystemExit(f"{path} does not fit a network with {core.obs_dim} inputs and {core.n_actions} actions. "
                         "Pass --trading if (and only if) it was trained with trading=True.") from e
    net.eval()
    return net, ckpt["iteration"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    # These two must match the CatanConfig the checkpoint was trained with.
    parser.add_argument("--trading", action="store_true")
    parser.add_argument("--max-offers", type=int, default=CONFIG.max_offers_per_turn)
    args = parser.parse_args()

    config = dataclasses.replace(CONFIG, trading=args.trading, max_offers_per_turn=args.max_offers)
    core = CatanCore(config)
    n = core.n_agents
    net, iteration = load_policy_net(args.checkpoint, core)
    print(f"{args.checkpoint} (iteration {iteration}), {args.games} games per match")

    print("1) one network vs 3 random bots")
    for k in range(n):
        seats = ["random"] * n
        seats[k] = make_policy(net, k, n)
        r = play_match(seats, args.games, config, seed=args.seed)
        bots = sum(r.win_rates[i] for i in range(n) if i != k)
        print(f"   network {k}: win rate {r.win_rates[k]:.2f} , three bots together {bots:.2f}")

    print("2) two networks vs 2 random bots")
    for a, b in itertools.combinations(range(n), 2):
        seats = ["random"] * n
        seats[a], seats[b] = make_policy(net, a, n), make_policy(net, b, n)
        r = play_match(seats, args.games, config, seed=args.seed)
        bots = sum(r.win_rates[i] for i in range(n) if i not in (a, b))
        print(f"   network {a} {r.win_rates[a]:.2f}, network {b} {r.win_rates[b]:.2f}, both bots together {bots:.2f}")

if __name__ == "__main__":
    main()