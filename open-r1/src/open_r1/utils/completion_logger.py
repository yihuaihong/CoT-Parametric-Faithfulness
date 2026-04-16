"""wandb Table callback for logging GRPO rollout completions.

Since TRL 0.14's GRPOTrainer does not upload completions to wandb, we
instead decorate each reward function so its last invocation's inputs
(prompts + completions) are cached on a module-global buffer. A small
TrainerCallback then flushes the buffer to a wandb.Table every N steps.

Usage in grpo.py:

    from open_r1.utils.completion_logger import (
        install_completion_capture, WandbCompletionTableCallback,
    )
    reward_funcs = install_completion_capture(reward_funcs)
    trainer = GRPOTrainer(
        ...,
        reward_funcs=reward_funcs,
        callbacks=[WandbCompletionTableCallback(every_n_steps=10,
                                                 max_rows_per_flush=8)],
    )
"""
from __future__ import annotations

import functools
from typing import Any, Callable

from transformers import TrainerCallback


_BUFFER: list[dict[str, Any]] = []  # module-global; flushed by callback


def install_completion_capture(reward_funcs: list[Callable]) -> list[Callable]:
    """Wrap every reward function so that on each call it appends the
    (prompt, completion, func_name, reward_value) rows to `_BUFFER`."""

    def _wrap(rf: Callable) -> Callable:
        name = getattr(rf, "__name__", "reward")

        @functools.wraps(rf)
        def wrapper(completions, *args, **kwargs):
            rewards = rf(completions, *args, **kwargs)
            try:
                prompts = kwargs.get("prompts") or kwargs.get("prompt") or [None] * len(completions)
                for i, (p, c, r) in enumerate(zip(prompts, completions, rewards)):
                    text = c[0]["content"] if isinstance(c, list) else str(c)
                    p_text = (p[-1]["content"] if isinstance(p, list) and p
                              else (p if isinstance(p, str) else ""))
                    _BUFFER.append({
                        "func": name, "idx": i,
                        "prompt": p_text, "completion": text,
                        "reward": float(r) if r is not None else 0.0,
                    })
            except Exception as e:
                _BUFFER.append({"func": name, "error": str(e)[:200]})
            return rewards

        return wrapper

    return [_wrap(rf) for rf in reward_funcs]


class WandbCompletionTableCallback(TrainerCallback):
    """Every `every_n_steps`, pop rows from the global buffer and log
    up to `max_rows_per_flush` to a wandb.Table keyed at the step."""

    def __init__(self, every_n_steps: int = 10, max_rows_per_flush: int = 8):
        self.every_n_steps = every_n_steps
        self.max_rows = max_rows_per_flush

    def on_log(self, args, state, control, logs=None, **kwargs):
        global _BUFFER
        step = state.global_step
        if step == 0 or step % self.every_n_steps != 0:
            return
        if not _BUFFER:
            return
        try:
            import wandb
        except ImportError:
            _BUFFER.clear()
            return
        if wandb.run is None:
            _BUFFER.clear()
            return

        # Take the LAST N rows (most recent rollouts).
        rows = _BUFFER[-self.max_rows:]
        _BUFFER.clear()

        table = wandb.Table(columns=["step", "func", "idx", "prompt",
                                     "completion", "reward"])
        for r in rows:
            table.add_data(
                step, r.get("func", "?"), r.get("idx", 0),
                (r.get("prompt") or "")[:500],
                (r.get("completion") or "")[:2000],
                r.get("reward", 0.0),
            )
        wandb.log({"train/completion_table": table}, step=step)
