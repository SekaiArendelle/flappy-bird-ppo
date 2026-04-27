from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch


class PPOInferenceAgent:
    """Inference-only PPO agent loader for Flappy Bird."""

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        self._device = torch.device(device)
        self._model = self._load_model(model_path).to(self._device)
        self._model.eval()

    def _load_model(self, model_path: Path) -> torch.nn.Module:
        if not model_path.exists():
            raise FileNotFoundError(f"PPO model file does not exist: {model_path}")

        try:
            return torch.jit.load(str(model_path), map_location=self._device)
        except Exception:
            checkpoint: Any = torch.load(str(model_path), map_location=self._device)
            if isinstance(checkpoint, torch.nn.Module):
                return checkpoint
            if isinstance(checkpoint, dict):
                if isinstance(checkpoint.get("model"), torch.nn.Module):
                    return checkpoint["model"]
                if isinstance(checkpoint.get("policy"), torch.nn.Module):
                    return checkpoint["policy"]

            raise ValueError(
                "Unsupported PPO model format. Provide a TorchScript model or a checkpoint "
                "containing a torch.nn.Module under key 'model' or 'policy'."
            )

    def reset(self) -> None:
        return None

    def act(self, observation) -> int:
        obs = np.asarray(observation, dtype=np.float32)
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(self._device)
        with torch.no_grad():
            output = self._model(obs_t)

        if isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        logits = torch.as_tensor(logits, device=self._device)
        if logits.ndim == 0:
            return int((torch.sigmoid(logits) > 0.5).item())
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        return int(torch.argmax(logits, dim=-1).item())
