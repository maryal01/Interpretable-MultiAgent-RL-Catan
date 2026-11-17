"""Play a trained checkpoint against bots and other checkpoints.

    python runeval.py checkpoints/iter_0199.pt --trading --seats this checkpoints/iter_0100.pt value value

--seats takes one entry per seat: "this" (the checkpoint), a bot name (random,
weighted, victory_point, value), or the path of another checkpoint trained with
the same settings. Without it: this random random random.
Each checkpoint holds one network per seat, so seat k is played by network k.
"""
import argparse
import dataclasses
import pathlib
import torch
from torchrl.modules import MultiAgentMLP
from catan_env import CatanConfig, CatanCore
from evaluate import BOTS, make_policy, play_match

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
    parser.add_argument("--seats", nargs="+", default=["this", "random", "random", "random"])
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    # These two must match the CatanConfig the checkpoints were trained with.
    parser.add_argument("--trading", action="store_true")
    parser.add_argument("--max-offers", type=int, default=CONFIG.max_offers_per_turn)
    args = parser.parse_args()

    config = dataclasses.replace(CONFIG, trading=args.trading, max_offers_per_turn=args.max_offers)
    core = CatanCore(config)
    n = core.n_agents
    if len(args.seats) != n:
        raise SystemExit(f"--seats needs {n} entries, one per seat; got {len(args.seats)}")

    nets, seats = {}, []
    for k, who in enumerate(args.seats):
        if who in BOTS:
            seats.append(who)
            continue
        path = args.checkpoint if who == "this" else who
        if path not in nets:
            nets[path], iteration = load_policy_net(path, core)
            print(f"{path} (iteration {iteration})")
        seats.append(make_policy(nets[path], k, n))

    labels = [pathlib.Path(who).stem[:13] for who in args.seats]
    print(play_match(seats, args.games, config, seed=args.seed, labels=labels).table())

if __name__ == "__main__":
    main()
