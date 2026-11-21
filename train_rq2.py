"""RQ2: can competing agents learn faster by periodically exchanging successful
strategies while keeping distinct policies?

Same IPPO setup as train.py (4 decentralised policies, shared critic, ClipPPO,
simple reward, no trading), plus ONE strategy-exchange condition chosen with
--exchange. Every condition gets the same frames, iterations and PPO settings, so
any difference in learning speed comes from the exchange itself.

  none        independent policies, no exchange. The baseline (same as train.py).
  shared      one policy network for all 4 agents. Maximum sharing, zero diversity:
              the upper bound on "exchange", the lower bound on "distinct".
  blend       weight-space exchange. Every --every iterations, each agent's weights
              move a fraction --alpha toward the best agent's weights.
  distill     behaviour-space exchange. Every --every iterations, each agent is
              trained to imitate the best agent's action choices on its own states.
  interleave  alternating centralised / decentralised policy training. Cycles of
              --decentral-iters independent iterations followed by --central-iters
              iterations in which every agent applies the SAME (averaged) update, so
              improvements are pooled but the agents' differences are preserved.

"Best agent" (blend, distill) = the one that won the most self-play games since
the last exchange, read from the collected batches (free). Ties are broken at random.

Logged every iteration to <out>/log.jsonl (cheap, from the training batch):
  wins per agent, games finished, frames per game, mean episode reward,
  js_diversity  mean pairwise Jensen-Shannon divergence between the 4 policies'
                action distributions on the same states (0 = identical behaviour)
  param_dist    mean pairwise L2 distance between agents' weights / mean weight norm
  phase         interleave only: "central" or "decentral"
  exchange      what was exchanged this iteration, if anything

Learning speed is measured offline with rq2_eval.py (win rate vs fixed bots per
checkpoint), because self-play reward cannot show it. run_rq2.sh runs the grid.

Checkpoints use train.py's format ("policy" with one network per agent, also for
`shared`), so runeval.py / profile_policies.py / distill.py work on them.

    python train_rq2.py --exchange none       --seed 0 --out results/rq2/runs/none_s0
    python train_rq2.py --exchange interleave --central-iters 5 --decentral-iters 15 --seed 0 \
                        --out results/rq2/runs/interleave_s0
    python train_rq2.py --exchange blend      --every 10 --alpha 0.3 --seed 0 --out results/rq2/runs/blend_s0
    python train_rq2.py --exchange distill    --every 10 --beta 1.0  --seed 0 --out results/rq2/runs/distill_s0
    python train_rq2.py --exchange shared     --seed 0 --out results/rq2/runs/shared_s0

Run with PYTHONHASHSEED=0 (run_rq2.sh does) for reproducible games: Catanatron's
move order depends on Python's per-process string hashing.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import torch
from tensordict.nn import TensorDictModule, set_composite_lp_aggregate
from torchrl.collectors import Collector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.modules import MaskedCategorical, MultiAgentMLP, ProbabilisticActor
from torchrl.objectives import ClipPPOLoss, ValueEstimators
from tqdm import tqdm

from catan_env import CatanConfig, CatanEnv
from rewardfn import make_reward_fn

OBS, MASK, ACTIVE = ("agents", "observation"), ("agents", "action_mask"), ("agents", "mask")


# ------------------------------------------------------------------ measurement helpers
def active_states(data, max_frames=None, generator=None):
    """Frames with exactly one active agent: (obs [M, D], legal-action mask [M, A], active agent [M])."""
    active = data.get(ACTIVE).reshape(data.batch_size.numel(), -1).bool()
    obs = data.get(OBS).reshape(active.shape[0], active.shape[1], -1)
    legal = data.get(MASK).reshape(active.shape[0], active.shape[1], -1).bool()
    keep = active.sum(-1) == 1
    idx = keep.nonzero().squeeze(-1)
    if max_frames is not None and idx.numel() > max_frames:
        idx = idx[torch.randperm(idx.numel(), generator=generator)[:max_frames]]
    who = active[idx].float().argmax(-1)
    return obs[idx, who], legal[idx, who], who


def all_policies_on(net, obs, legal):
    """Log-probs of every agent's policy on the same states: [M, n_agents, A].
    MultiAgentMLP runs network k on input slot k, so putting the same observation in
    every slot evaluates all 4 networks on it in one forward pass."""
    n = net.n_agents
    logits = net(obs.unsqueeze(1).expand(-1, n, -1))
    logits = logits.masked_fill(~legal.unsqueeze(1), -1e9)
    return torch.log_softmax(logits, -1)


def js_diversity(net, obs, legal) -> float:
    """Mean pairwise Jensen-Shannon divergence between the agents' policies (nats, 0..ln 2)."""
    with torch.no_grad():
        logp = all_policies_on(net, obs, legal)
    p = logp.exp()
    vals = []
    for a, b in itertools.combinations(range(net.n_agents), 2):
        m = 0.5 * (p[:, a] + p[:, b])
        logm = torch.log(m.clamp_min(1e-12))
        js = 0.5 * (p[:, a] * (logp[:, a] - logm)).sum(-1) + 0.5 * (p[:, b] * (logp[:, b] - logm)).sum(-1)
        vals.append(js.mean())
    return max(0.0, float(torch.stack(vals).mean()))


def param_distance(net, shared: bool) -> float:
    """Mean pairwise L2 distance between agents' weight vectors, relative to their mean norm."""
    if shared:
        return 0.0
    flat = torch.cat([p.detach().reshape(net.n_agents, -1) for p in net.params.values(True, True)], -1)
    d = torch.cdist(flat, flat)
    n = net.n_agents
    return float(d.sum() / (n * (n - 1)) / flat.norm(dim=-1).mean())


def batch_outcomes(data, n_agents):
    """Wins per agent, games finished, and frames per game in this batch."""
    reward = data.get(("next", "agents", "reward")).reshape(-1, n_agents)
    term = data.get(("next", "terminated")).reshape(-1)
    done = data.get(("next", "done")).reshape(-1)
    ended = reward[term]
    winners = ended.argmax(-1)[ended.max(-1).values > 0]   # truncated games have no winner
    wins = torch.bincount(winners, minlength=n_agents)
    games = int(done.sum())
    return wins, games, (data.batch_size.numel() / games if games else float("nan"))


# ------------------------------------------------------------------ exchange strategies
# shared: nothing to do at run time. The policy is built with share_params=True, so
# all four agents literally use one network and every update trains it on all seats'
# experience. Diversity is zero by construction.


def blend_toward(net, best: int, alpha: float):
    """blend: theta_i <- (1 - alpha) * theta_i + alpha * theta_best for every agent i.

    Copies part of the most successful agent's weights into everyone else. alpha = 1
    would clone the best agent (diversity collapses); small alpha keeps agents
    distinct but nudges them toward what is currently winning. The best agent itself
    is unchanged (moving toward itself is a no-op). Adam's moment estimates are left
    as they are."""
    with torch.no_grad():
        for p in net.params.values(True, True):          # each p is [n_agents, ...]
            target = p.data[best].clone()
            p.data.lerp_(target.expand_as(p.data), alpha)


def distill_toward(net, best, replay_buffer, optim, n_steps, beta, max_grad_norm):
    """distill: minimise beta * KL(pi_best(.|s) || pi_i(.|s)) for every agent i != best.

    Each agent is trained to choose actions like the best agent, but only on states the
    agent itself visited, and only for a few gradient steps. Its weights stay its own,
    so it keeps whatever it does differently where the teacher has no opinion. The
    teacher's output is detached, so the best agent is not changed. Returns mean KL."""
    params = list(net.parameters())
    kls = []
    for _ in range(n_steps):
        obs, legal, who = active_states(replay_buffer.sample())
        keep = who != best
        if not keep.any():
            continue
        obs, legal, who = obs[keep], legal[keep], who[keep]
        logp = all_policies_on(net, obs, legal)
        rows = torch.arange(obs.shape[0])
        student, teacher = logp[rows, who], logp[:, best].detach()
        kl = (teacher.exp() * (teacher - student)).sum(-1).mean()
        (beta * kl).backward()
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optim.step()
        optim.zero_grad()
        kls.append(float(kl))
    return sum(kls) / len(kls) if kls else float("nan")


def interleave_phase(i: int, central_iters: int, decentral_iters: int) -> str:
    """interleave schedule: each cycle is `decentral_iters` independent iterations, then
    `central_iters` centralised ones. Starting decentralised gives the first central
    phase a window of games from which to measure who is winning (for --central-weighting wins)."""
    return "central" if i % (central_iters + decentral_iters) >= decentral_iters else "decentral"


def interleave_weights(window_wins: torch.Tensor, mode: str) -> torch.Tensor:
    """How much each agent's update counts in a centralised phase.
    mean: all agents equally (plain pooled training).
    wins: proportional to wins in the preceding decentralised phase, add-one smoothed so
          an agent that won nothing still contributes a little."""
    n = window_wins.numel()
    if mode == "mean" or window_wins.sum() == 0:
        return torch.full((n,), 1.0 / n)
    w = window_wins.float() + 1.0
    return w / w.sum()


def snapshot(net):
    return [p.detach().clone() for p in net.params.values(True, True)]


def apply_common_update(net, before, weights):
    """interleave (central phase): replace each agent's own update with the weighted
    average of all agents' updates.

    Called right after optim.step(). Each agent's PPO gradient comes from its own
    seats' experience, so averaging the updates is (up to Adam's per-agent scaling)
    training one shared policy on the pooled experience of all agents: improvements
    found by any agent reach every agent. Because every agent receives the identical
    change, the differences between agents (theta_i - mean theta) are preserved
    exactly: centralised phases share learning without erasing individuality, and
    diversity can only grow in the decentralised phases."""
    with torch.no_grad():
        for p, b in zip(net.params.values(True, True), before):
            delta = p.data - b                                         # [n_agents, ...]
            w = weights.view(-1, *([1] * (delta.dim() - 1)))
            common = (w * delta).sum(0, keepdim=True)
            p.data.copy_(b + common.expand_as(delta))


def export_policy(net, shared: bool, n_agents: int) -> dict:
    """train.py's checkpoint format: every parameter has a leading agent dimension."""
    sd = net.state_dict()
    if shared:
        sd = {k: v.unsqueeze(0).expand(n_agents, *v.shape).clone() for k, v in sd.items()}
    return sd


# ------------------------------------------------------------------ training
def train(args):
    torch.manual_seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    gen = torch.Generator().manual_seed(args.seed)
    device = torch.device("cpu")
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=1)

    num_epochs, minibatch_size, lr, max_grad_norm = 10, min(400, args.frames), 3e-4, 1.0
    clip_epsilon, gamma, lmbda, entropy_eps = 0.2, 0.999, 0.9, 0.01
    set_composite_lp_aggregate(False).set()

    env = CatanEnv(CatanConfig(num_players=4, map_type="BASE", vps_to_win=8, trading=args.trading),
                   seed=args.seed, device=device, reward_fn=make_reward_fn(args.reward))
    env = TransformedEnv(env, RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]))
    n_agents = env.core.n_agents
    n_obs = env.observation_spec[OBS].shape[-1]
    n_actions = env.full_action_spec[env.action_key].space.n
    shared = args.exchange == "shared"

    policy_net = MultiAgentMLP(n_agent_inputs=n_obs, n_agent_outputs=n_actions, n_agents=n_agents,
                               centralised=False, share_params=shared, device=device, depth=2,
                               num_cells=256, activation_class=torch.nn.Tanh)
    policy = ProbabilisticActor(
        module=TensorDictModule(policy_net, in_keys=[OBS], out_keys=[("agents", "logits")]),
        in_keys={"logits": ("agents", "logits"), "mask": MASK},
        out_keys=[env.action_key], distribution_class=MaskedCategorical, return_log_prob=True)
    critic_net = MultiAgentMLP(n_agent_inputs=n_obs, n_agent_outputs=1, n_agents=n_agents,
                               centralised=False, share_params=True, device=device, depth=2,
                               num_cells=256, activation_class=torch.nn.Tanh)
    critic = TensorDictModule(critic_net, in_keys=[OBS], out_keys=[("agents", "state_value")])

    collector = Collector(env, policy, device=device, storing_device=device, frames_per_batch=args.frames,
                          total_frames=args.frames * args.iters, auto_register_policy_transforms=True)
    replay_buffer = ReplayBuffer(storage=LazyTensorStorage(args.frames, device=device),
                                 sampler=SamplerWithoutReplacement(), batch_size=minibatch_size)
    loss_module = ClipPPOLoss(actor_network=policy, critic_network=critic, clip_epsilon=clip_epsilon,
                              entropy_coeff=entropy_eps, normalize_advantage=False)
    loss_module.set_keys(reward=env.reward_key, action=env.action_key, value=("agents", "state_value"),
                         done=("agents", "done"), terminated=("agents", "terminated"))
    loss_module.loss_mask_key = ACTIVE
    loss_module.make_value_estimator(ValueEstimators.GAE, gamma=gamma, lmbda=lmbda)
    GAE = loss_module.value_estimator
    optim = torch.optim.Adam(loss_module.parameters(), lr)

    def save(i, final=False):
        ckpt = {"iteration": i, "frames": (i + 1) * args.frames, "args": vars(args),
                "policy": export_policy(policy_net, shared, n_agents), "critic": critic_net.state_dict()}
        if final:
            ckpt["optimizer"] = optim.state_dict()
        torch.save(ckpt, os.path.join(args.out, f"iter_{i:04d}.pt") if i >= 0
                   else os.path.join(args.out, "iter_before_training.pt"))

    save(-1)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    window_wins = torch.zeros(n_agents, dtype=torch.long)   # wins since the last exchange / phase change
    central_weights = None
    pbar = tqdm(total=args.iters)
    for i, data in enumerate(collector):
        t0 = time.time()
        wins, games, frames_per_game = batch_outcomes(data, n_agents)
        done = data.get(("next", "agents", "done"))
        ep_rew = data.get(("next", "agents", "episode_reward"))[done]
        exchange, distill_kl = None, None

        # interleave: decide this iteration's phase. Entering a central phase fixes the
        # agents' weights for the whole phase from the preceding decentral phase's wins.
        phase = None
        if args.exchange == "interleave":
            phase = interleave_phase(i, args.central_iters, args.decentral_iters)
            if phase == "central" and central_weights is None:
                central_weights = interleave_weights(window_wins, args.central_weighting)
                exchange = {"central_weights": [round(float(w), 3) for w in central_weights],
                            "window_wins": window_wins.tolist()}
                window_wins.zero_()
            elif phase == "decentral":
                central_weights = None
        if phase != "central":   # interleave weights use decentral-phase wins only
            window_wins += wins

        # ---- PPO update (identical for every condition)
        with torch.no_grad():
            GAE(data, params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params)
        replay_buffer.extend(data.reshape(-1))
        stats = []
        for _ in range(num_epochs):
            for _ in range(args.frames // minibatch_size):
                lv = loss_module(replay_buffer.sample())
                loss = lv["loss_objective"] + lv["loss_critic"] + lv["loss_entropy"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                before = snapshot(policy_net) if phase == "central" else None
                optim.step()
                if before is not None:                     # interleave, central phase
                    apply_common_update(policy_net, before, central_weights)
                optim.zero_grad()
                stats.append([float(lv[k]) for k in ("loss_objective", "loss_critic", "loss_entropy")])

        # ---- periodic exchange toward the best agent (blend, distill)
        if args.exchange in ("blend", "distill") and (i + 1) % args.every == 0 and window_wins.sum() > 0:
            top = (window_wins == window_wins.max()).nonzero().squeeze(-1)
            best = int(top[torch.randint(len(top), (1,), generator=gen)])   # random tie-break
            if args.exchange == "blend":
                blend_toward(policy_net, best, args.alpha)
            else:
                steps = args.distill_epochs * (args.frames // minibatch_size)
                distill_kl = distill_toward(policy_net, best, replay_buffer, optim, steps, args.beta, max_grad_norm)
            exchange = {"best": best, "window_wins": window_wins.tolist()}
            window_wins.zero_()
        collector.update_policy_weights_()

        # ---- logging
        obs, legal, _ = active_states(data, max_frames=2048, generator=gen)
        mean_stats = [sum(s) / len(stats) for s in zip(*stats)]
        record = {
            "iter": i, "frames": (i + 1) * args.frames, "wins": wins.tolist(), "games": games,
            "frames_per_game": frames_per_game,
            "episode_reward": float(ep_rew.mean()) if ep_rew.numel() else None,
            "js_diversity": js_diversity(policy_net, obs, legal),
            "param_dist": param_distance(policy_net, shared),
            "loss_objective": mean_stats[0], "loss_critic": mean_stats[1], "loss_entropy": mean_stats[2],
            "phase": phase, "exchange": exchange, "distill_kl": distill_kl,
            "update_seconds": time.time() - t0,
        }
        log.write(json.dumps(record) + "\n")
        log.flush()
        desc = f"js={record['js_diversity']:.3f} games={games}"
        if phase:
            desc += f" {phase}"
        elif exchange:
            desc += f" exch->{exchange['best']}"
        pbar.set_description(desc)
        pbar.update()

        if i % args.save_every == 0 or i == args.iters - 1:
            save(i, final=i == args.iters - 1)
    log.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--exchange", choices=["none", "shared", "blend", "distill", "interleave"], default="none")
    # blend / distill
    p.add_argument("--every", type=int, default=10, help="blend/distill: iterations between exchanges")
    p.add_argument("--alpha", type=float, default=0.3, help="blend: fraction moved toward the best agent")
    p.add_argument("--beta", type=float, default=1.0, help="distill: weight of the KL loss")
    p.add_argument("--distill-epochs", type=int, default=2, help="distill: passes over the batch per exchange")
    # interleave
    p.add_argument("--central-iters", type=int, default=5, help="interleave: iterations per centralised phase")
    p.add_argument("--decentral-iters", type=int, default=15, help="interleave: iterations per decentralised phase")
    p.add_argument("--central-weighting", choices=["mean", "wins"], default="mean",
                   help="interleave: average updates equally, or weighted by recent wins")
    # common
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--frames", type=int, default=6_000)
    p.add_argument("--reward", default="simple")
    p.add_argument("--trading", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--threads", type=int, default=None, help="torch threads (set when running several jobs)")
    p.add_argument("--out", required=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
