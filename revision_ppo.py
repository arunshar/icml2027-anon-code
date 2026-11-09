"""PPO losses and a learned value function for terminal contextual bandits.

Each action is one complete response to a fixed training prompt. Its terminal
reward is its return, and its advantage is reward minus the pre-rollout value.
The critic sees only the prompt index. It cannot inspect the sampled response.

The actor uses the clipped objective in Schulman et al. (2017), Equation 7.
The critic uses the squared return error in Equation 9. Multiple optimization
epochs reuse the frozen behavior log-probabilities and advantages, as required
by Algorithm 1. The definitions are also documented by OpenAI Spinning Up.

Sources:
https://arxiv.org/pdf/1707.06347
https://spinningup.openai.com/en/latest/algorithms/ppo.html

The caller must snapshot values before sampling or scoring the current batch.
The caller must score the same distribution that generated the responses.
Full-model token log-probabilities require untruncated, unit-temperature
sampling without a repetition penalty. The caller owns reward scaling,
optimizers, minibatches, and repeated optimization epochs. These functions
never centre or normalize advantages and never modify historical trainers.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


DEFAULT_CLIP_COEF = 0.2
DEFAULT_OPTIMIZATION_EPOCHS = 4
DEFAULT_CRITIC_LR = 0.3


def _vector(name: str, value: Tensor, *, floating: bool = True) -> None:
    """Validate a nonempty vector before any broadcasting can hide a mistake."""
    if not isinstance(value, Tensor) or value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional tensor")
    if floating:
        if not value.is_floating_point():
            raise ValueError(f"{name} must have a floating-point dtype")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")


def _prompt_ids(prompt_ids: Tensor) -> None:
    _vector("prompt_ids", prompt_ids, floating=False)
    if prompt_ids.dtype != torch.long:
        raise ValueError("prompt_ids must have dtype torch.long")
    if (prompt_ids < 0).any().item():
        raise ValueError("prompt_ids must be nonnegative")


class PromptValueCritic(nn.Module):
    """Learn one expected return per fixed training prompt by regression.

    A scalar for each context can represent any state-value function on this
    finite prompt set. The caller should not allocate entries for held-out
    evaluation prompts. Actor evaluation does not require a critic prediction.
    The zero initialization consumes no policy sampling random numbers.
    """

    def __init__(self, n_prompts: int):
        super().__init__()
        if isinstance(n_prompts, bool) or not isinstance(n_prompts, int) or n_prompts < 1:
            raise ValueError("n_prompts must be a positive integer")
        self.n_prompts = n_prompts
        # Supplying the weights avoids Embedding's random initialization.
        self.values = nn.Embedding(n_prompts, 1, _weight=torch.zeros(n_prompts, 1))

    def forward(self, prompt_ids: Tensor) -> Tensor:
        _prompt_ids(prompt_ids)
        if (prompt_ids >= self.n_prompts).any().item():
            raise ValueError("prompt_ids includes a prompt outside the training set")
        if prompt_ids.device != self.values.weight.device:
            raise ValueError("prompt_ids and the critic must be on the same device")
        return self.values(prompt_ids).squeeze(-1)


@torch.no_grad()
def snapshot_values(critic: PromptValueCritic, prompt_ids: Tensor) -> Tensor:
    """Copy value predictions before the caller generates or scores responses."""
    return critic(prompt_ids).detach().clone()


@dataclass(frozen=True)
class FrozenBanditBatch:
    """Hold copied rollout data that remains fixed across optimization epochs."""

    prompt_ids: Tensor
    behavior_log_probs: Tensor
    returns: Tensor
    old_values: Tensor
    advantages: Tensor

    def __len__(self) -> int:
        return self.prompt_ids.numel()

    def select(self, indices: Tensor) -> "FrozenBanditBatch":
        """Select a minibatch without recomputing its values or advantages."""
        _prompt_ids(indices)
        if indices.device != self.prompt_ids.device:
            raise ValueError("indices and the rollout must be on the same device")
        if (indices >= len(self)).any().item():
            raise ValueError("indices includes an entry outside the rollout")
        return FrozenBanditBatch(
            self.prompt_ids[indices], self.behavior_log_probs[indices],
            self.returns[indices], self.old_values[indices], self.advantages[indices],
        )


def freeze_bandit_batch(
    prompt_ids: Tensor,
    behavior_log_probs: Tensor,
    rewards: Tensor,
    old_values: Tensor,
) -> FrozenBanditBatch:
    """Freeze terminal returns and pre-rollout value estimates for PPO.

    The caller must supply values from snapshot_values, taken before it sees
    current rewards. The signature accepts no critic that could be refitted
    here. Rewards, values, and old probabilities are cloned as well as detached.
    The actor cannot backpropagate into the critic, reward, or behavior policy.
    """
    _prompt_ids(prompt_ids)
    for name, tensor in (("behavior_log_probs", behavior_log_probs),
                         ("rewards", rewards), ("old_values", old_values)):
        _vector(name, tensor)
        if tensor.shape != prompt_ids.shape:
            raise ValueError(f"{name} and prompt_ids must have the same shape")
        if tensor.device != prompt_ids.device:
            raise ValueError(f"{name} and prompt_ids must be on the same device")
    if behavior_log_probs.dtype != rewards.dtype or rewards.dtype != old_values.dtype:
        raise ValueError("probabilities, rewards, and old_values must share a dtype")
    fixed_rewards = rewards.detach().clone()
    fixed_values = old_values.detach().clone()
    fixed_advantages = fixed_rewards - fixed_values
    _vector("advantages", fixed_advantages)
    return FrozenBanditBatch(
        prompt_ids.detach().clone(), behavior_log_probs.detach().clone(),
        fixed_rewards, fixed_values, fixed_advantages,
    )


@dataclass(frozen=True)
class PPOActorLoss:
    """Return the differentiable loss and detached surrogate diagnostics."""

    loss: Tensor
    ratios: Tensor
    clip_fraction: Tensor
    active_clip_fraction: Tensor
    approximate_kl: Tensor

    def metrics(self) -> dict[str, float]:
        """Convert diagnostics to scalars when the caller records an epoch."""
        return {
            "policy_loss": float(self.loss.detach().item()),
            "clip_fraction": float(self.clip_fraction.item()),
            "active_clip_fraction": float(self.active_clip_fraction.item()),
            "approximate_behavior_kl": float(self.approximate_kl.item()),
            "ratio_min": float(self.ratios.min().item()),
            "ratio_max": float(self.ratios.max().item()),
        }


def ppo_actor_loss(
    new_log_probs: Tensor,
    batch: FrozenBanditBatch,
    clip_coef: float = DEFAULT_CLIP_COEF,
) -> PPOActorLoss:
    """Compute the completion-level PPO clipped surrogate loss.

    clip_fraction counts ratios outside the clip interval. active_clip_fraction
    counts samples where the clipped branch actually limits the objective.
    A ratio outside the interval can remain unclipped when it worsens the
    objective. The approximate KL diagnostic compares the frozen behavior
    policy with the current actor on behavior samples. It is not an exact KL.
    """
    _vector("new_log_probs", new_log_probs)
    if not math.isfinite(clip_coef) or not 0 < clip_coef < 1:
        raise ValueError("clip_coef must lie strictly between zero and one")
    if new_log_probs.shape != batch.behavior_log_probs.shape:
        raise ValueError("new_log_probs and the rollout must have the same shape")
    if (new_log_probs.device != batch.behavior_log_probs.device
            or new_log_probs.dtype != batch.behavior_log_probs.dtype):
        raise ValueError("new_log_probs and the rollout must share device and dtype")
    old_log_probs = batch.behavior_log_probs.detach()
    advantages = batch.advantages.detach()
    log_ratio = new_log_probs - old_log_probs
    ratios = log_ratio.exp()
    if not torch.isfinite(ratios).all().item():
        raise FloatingPointError("PPO probability ratio overflowed")
    unclipped = ratios * advantages
    clipped = ratios.clamp(1 - clip_coef, 1 + clip_coef) * advantages
    loss = -torch.minimum(unclipped, clipped).mean()
    if not torch.isfinite(loss).item():
        raise FloatingPointError("PPO surrogate loss is not finite")
    with torch.no_grad():
        outside = (ratios < 1 - clip_coef) | (ratios > 1 + clip_coef)
        active = ((advantages > 0) & (ratios > 1 + clip_coef)
                  | (advantages < 0) & (ratios < 1 - clip_coef))
        approximate_kl = ((ratios - 1) - log_ratio).mean()
    return PPOActorLoss(
        loss, ratios.detach(), outside.to(ratios.dtype).mean(),
        active.to(ratios.dtype).mean(), approximate_kl,
    )


def value_loss(predicted_values: Tensor, batch: FrozenBanditBatch) -> Tensor:
    """Fit the critic to observed terminal returns with mean squared error.

    There is no bootstrap target in this terminal bandit. The caller should
    optimize only the critic's parameters with this loss. Its reward target
    is fixed, and fitting it never changes the actor's stored advantages.
    """
    _vector("predicted_values", predicted_values)
    if predicted_values.shape != batch.returns.shape:
        raise ValueError("predicted_values and returns must have the same shape")
    if (predicted_values.device != batch.returns.device
            or predicted_values.dtype != batch.returns.dtype):
        raise ValueError("predicted_values and returns must share device and dtype")
    return (predicted_values - batch.returns.detach()).square().mean()
