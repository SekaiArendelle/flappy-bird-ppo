# flappy-bird-ppo

This project uses `black` for Python code formatting.

## Use the `agent` module

Run from the project root:

```bash
python -m agent --help
```

### 1. DIY (rule-based) agent

```bash
python -m agent --agent diy --episodes 3 --render
```

### 2. PPO agent

```bash
python -m agent --agent ppo --model-path path\to\model.pt --episodes 3 --render
```

### 3. Train PPO (Actor-Critic + CNN)

```bash
python -m train
```

The trainer uses DIY-guided engineered features and reward shaping, and also adds a
behavior-cloning auxiliary loss from the DIY policy as a baseline signal.
Imitation is enabled in early training, and is automatically disabled once
`mean_score_10` is close to the measured DIY baseline score
(`--imitation-stop-score-gap`), after which optimization relies on PPO losses and
reward shaping only.
Training runs indefinitely until interrupted by `Ctrl+C`.
Training writes two checkpoints to `checkpoints\`:
`latest.pt` (latest update) and `best.pt` (best `mean_score_10`,
with `mean_reward_10` as tie-breaker).

For full trainer CLI arguments and descriptions, run:

```bash
python -m train --help
```
