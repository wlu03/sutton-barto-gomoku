"""Multi-run testbed harness for Ch. 2 figures.

For each independent run: spawn a fresh testbed problem, a fresh agent,
play n_steps pulls, record reward and whether the pulled arm was optimal.
The notebook averages across runs to get the canonical curves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from gomoku.agents.bandits.base import BanditAgent
from gomoku.agents.bandits.testbed import BanditTestbed



AgentFactory = Callable[[np.random.Generator], BanditAgent]

@dataclass
class ExperimentResult:
    """Per-run x per-step records. Average over axis=0 for the standard curves."""
    rewards: np.ndarray  # (n_runs, n_steps) float
    optimal: np.ndarray  # (n_runs, n_steps) bool

    @property
    def mean_reward(self) -> np.ndarray:    # (n_steps,)
        return self.rewards.mean(axis=0)

    @property
    def optimal_pct(self) -> np.ndarray:    # (n_steps,)
        return self.optimal.mean(axis=0)


def run_experiment(
    agent_factory: AgentFactory,
    *,
    n_arms: int = 10,
    n_steps: int = 1000,
    n_runs: int = 2000,
    nonstationary: bool = False,
    base_seed: int = 0,
    progress: bool = False,
) -> ExperimentResult:
    """Run `agent_factory` on `n_runs` independent testbed problems."""
    rewards = np.zeros((n_runs, n_steps), dtype=np.float64)
    optimal = np.zeros((n_runs, n_steps), dtype=bool)

    runs = range(n_runs)
    if progress:
        try:
            from tqdm import tqdm
            runs = tqdm(runs, desc="runs")
        except ImportError:
            pass  # tqdm is optional; degrade silently

    for run in runs:
        ss = np.random.SeedSequence((base_seed, run))
        bed_rng, agent_rng = (np.random.default_rng(s) for s in ss.spawn(2))

        testbed = BanditTestbed(
            n_arms=n_arms, rng=bed_rng, nonstationary=nonstationary
        )
        agent = agent_factory(agent_rng)

        for t in range(n_steps):
            a = agent.select_action()
            optimal[run, t] = (a == testbed.optimal_action)
            r = testbed.step(a)
            agent.update(a, r)
            rewards[run, t] = r

    return ExperimentResult(rewards=rewards, optimal=optimal)


# ──────────────────────────────────────────────────────────────────────
# CLI: `python gomoku/agents/bandits/runner.py`
# Runs the canonical Fig 2.2-style comparison and writes a PNG.
# ──────────────────────────────────────────────────────────────────────

def _default_configs() -> dict[str, AgentFactory]:
    from gomoku.agents.bandits.epsilon_greedy import EpsilonGreedyAgent
    from gomoku.agents.bandits.gradient import GradientAgent
    from gomoku.agents.bandits.ucb import UCBAgent
    return {
        "greedy (e=0)":      lambda rng: EpsilonGreedyAgent(10, epsilon=0.0, rng=rng),
        "e=0.01":            lambda rng: EpsilonGreedyAgent(10, epsilon=0.01, rng=rng),
        "e=0.1":             lambda rng: EpsilonGreedyAgent(10, epsilon=0.1, rng=rng),
        "optimistic e=0":    lambda rng: EpsilonGreedyAgent(10, epsilon=0.0, alpha=0.1, q_init=5.0, rng=rng),
        "UCB c=2":           lambda rng: UCBAgent(10, c=2.0, rng=rng),
        "gradient a=0.1":    lambda rng: GradientAgent(10, alpha=0.1, rng=rng),
    }


def _print_table(results: dict[str, ExperimentResult]) -> None:
    print()
    print(f"{'agent':<22} {'mean R (last 100)':>20} {'optimal % (last 100)':>22}")
    print("-" * 66)
    for name, r in results.items():
        mr = float(r.mean_reward[-100:].mean())
        op = float(r.optimal_pct[-100:].mean()) * 100
        print(f"{name:<22} {mr:>20.3f} {op:>21.1f}%")


def _plot(results: dict[str, ExperimentResult], out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_r, ax_p) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for name, r in results.items():
        ax_r.plot(r.mean_reward, label=name, linewidth=1.2)
        ax_p.plot(r.optimal_pct * 100, label=name, linewidth=1.2)

    ax_r.set_ylabel("average reward")
    ax_r.grid(True, alpha=0.3)
    ax_r.legend(loc="lower right", fontsize=9)
    ax_r.set_title("Sutton & Barto Ch. 2 — 10-armed testbed")

    ax_p.set_ylabel("% optimal action")
    ax_p.set_xlabel("step")
    ax_p.set_ylim(0, 100)
    ax_p.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    print(f"\nplot -> {out_path}")


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run the Ch. 2 bandit testbed comparison.")
    p.add_argument("--n-runs",  type=int, default=2000, help="independent runs per agent")
    p.add_argument("--n-steps", type=int, default=1000, help="pulls per run")
    p.add_argument("--out",     default="bandits.png",  help="output PNG path")
    p.add_argument("--no-plot", action="store_true",    help="skip plotting (just print table)")
    p.add_argument("--show",    action="store_true",    help="also open an interactive window")
    args = p.parse_args()

    configs = _default_configs()
    results: dict[str, ExperimentResult] = {}
    for name, factory in configs.items():
        print(f"[{name}]  running {args.n_runs} runs x {args.n_steps} steps …")
        results[name] = run_experiment(
            factory, n_runs=args.n_runs, n_steps=args.n_steps, progress=True,
        )

    _print_table(results)

    if not args.no_plot:
        _plot(results, args.out)
        if args.show:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == "__main__":
    _main()
