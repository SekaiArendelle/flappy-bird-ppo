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

For full CLI arguments and descriptions, run:

```bash
python -m agent --help
```
