# sutton-barto-gomoku

> Implementing every major algorithm from Sutton & Barto's *Reinforcement Learning: An Introduction* (2nd ed.), with Gomoku as the through-line — from ε-greedy bandits to an AlphaZero-style self-play agent.

<p align="center">
  <img src="https://images.unsplash.com/photo-1633974026122-2861ae7b6087?w=1200&h=400&fit=crop" alt="Gomoku board" width="720"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"/>
  <img src="https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg" alt="PyTorch 2.x"/>
  <img src="https://img.shields.io/github/license/YOUR_USERNAME/sutton-barto-gomoku" alt="License"/>
  <img src="https://img.shields.io/github/actions/workflow/status/YOUR_USERNAME/sutton-barto-gomoku/ci.yml?branch=main" alt="CI"/>
  <img src="https://img.shields.io/github/last-commit/YOUR_USERNAME/sutton-barto-gomoku" alt="Last commit"/>
  <img src="https://img.shields.io/badge/book-Sutton%20%26%20Barto-success" alt="Sutton & Barto"/>
</p>

---

## About

This repo is a long-form study project: one game, every algorithm. I work through *Reinforcement Learning: An Introduction* chapter by chapter and implement each method against a shared Gomoku environment, scaling the board size as the methods get more powerful.

The goal isn't to build the strongest possible Gomoku agent — it's to deeply understand *why* each algorithm exists, what it's trying to fix about the previous one, and where it breaks. Every agent here can play against every other one, so the progression is measurable, not just narrative.

## The arc

| Part | Chapters | Board | Methods |
|------|----------|-------|---------|
| I — Tabular | 2–8 | 6×6 | Bandits, DP, MC, TD, n-step, Dyna-Q, MCTS | 
| II — Approximate | 9–13 | 9×9 | Linear FA, semi-gradient TD, TD(λ), REINFORCE, A2C |
| III — Deep | 13+ / Ch. 16 | 15×15 | DQN, PPO, AlphaZero-style |

## Why Gomoku

Five-in-a-row scales cleanly from tiny boards (where tabular DP is tractable) to full 15×15 (where only deep methods work). The spatial structure rewards CNNs, the branching factor gives MCTS something real to chew on, and self-play removes the need for a fixed opponent. It's the same logic that made it a popular small-scale AlphaZero target.

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/sutton-barto-gomoku
cd sutton-barto-gomoku
pip install -e .

# Play a human vs. random baseline game
python -m gomoku.play --black human --white random --size 9

# Train a tabular Q-learning agent via self-play
python -m gomoku.train qlearning --size 6 --episodes 100000

# Watch two trained agents play
python -m gomoku.play --black checkpoints/qlearn.pt --white checkpoints/sarsa.pt
```

## Repo layout

```
gomoku/
  env/              # Gym-style Gomoku environment (shared by every agent)
  agents/
    bandits/        # Ch. 2
    tabular/        # Ch. 4–8
    linear/         # Ch. 9–13
    deep/           # DQN, PPO, AlphaZero
  tournament/       # Round-robin evaluation between agent versions
notebooks/          # One per chapter, with experiments and writeups
docs/
  chapters/         # Notes on each chapter: what the method is, what broke, what helped
```

## Results

Cross-play results updated as agents are added. All numbers from 1000-game round-robin self-play on the noted board size.

| | Random | ε-greedy | Q-learning | Dyna-Q | TD(λ) | REINFORCE | AlphaZero |
|---|---|---|---|---|---|---|---|
| Random | — | | | | | | |
| ε-greedy | | — | | | | | |
| Q-learning | | | — | | | | |
| *(filled in as agents are built)* | | | | | | | |

## Writeups

Each chapter has a notes file in [`docs/chapters/`](docs/chapters/) covering: what the algorithm is, what I expected to happen, what actually happened, and what surprised me. These are the closest thing to a textbook of my own.

## Acknowledgments

Built while reading Sutton & Barto's [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html) (2nd ed., MIT Press). Architectural debts to Junxiao Song's AlphaZero-Gomoku and DeepMind's AlphaZero paper.

## License

MIT
