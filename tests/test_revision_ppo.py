"""Check PPO's frozen baseline, learned values, and active clipped objective."""
import pytest

torch = pytest.importorskip("torch")

from revision_ppo import (  # noqa: E402
    DEFAULT_CRITIC_LR,
    PromptValueCritic,
    freeze_bandit_batch,
    ppo_actor_loss,
    snapshot_values,
    value_loss,
)


def _batch(rewards, *, old_values=None, old_log_probs=None, prompt_ids=None):
    rewards = torch.as_tensor(rewards, dtype=torch.float64)
    if old_values is None:
        old_values = torch.zeros_like(rewards)
    if old_log_probs is None:
        old_log_probs = torch.zeros_like(rewards)
    if prompt_ids is None:
        prompt_ids = torch.zeros(rewards.numel(), dtype=torch.long)
    return freeze_bandit_batch(prompt_ids, old_log_probs, rewards, old_values)


def test_clipped_surrogate_uses_the_correct_branch_for_each_advantage_sign():
    batch = _batch([1.0, 1.0, -1.0, -1.0])
    log_probs = torch.tensor([1.5, 0.5, 1.5, 0.5], dtype=torch.float64).log()
    log_probs.requires_grad_()
    result = ppo_actor_loss(log_probs, batch)
    # The improving positive and negative cases are limited to 1.2 and 0.8.
    assert result.loss.item() == pytest.approx(-(1.2 + 0.5 - 1.5 - 0.8) / 4)
    assert result.clip_fraction.item() == 1.0
    assert result.active_clip_fraction.item() == 0.5
    result.loss.backward()
    torch.testing.assert_close(
        log_probs.grad, torch.tensor([0.0, -0.125, 0.375, 0.0], dtype=torch.float64),
    )


def test_frozen_rollout_copies_inputs_and_blocks_gradients_to_past_quantities():
    ids = torch.tensor([0, 0], dtype=torch.long)
    old = torch.tensor([-1.0, -2.0], dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([1.0, -1.0], dtype=torch.float64, requires_grad=True)
    values = torch.tensor([0.25, 0.25], dtype=torch.float64, requires_grad=True)
    batch = freeze_bandit_batch(ids, old, rewards, values)
    with torch.no_grad():
        ids.fill_(1)
        old.add_(10)
        rewards.add_(10)
        values.add_(10)
    torch.testing.assert_close(batch.prompt_ids, torch.tensor([0, 0]))
    torch.testing.assert_close(batch.behavior_log_probs, torch.tensor([-1., -2.], dtype=torch.float64))
    torch.testing.assert_close(batch.advantages, torch.tensor([0.75, -1.25], dtype=torch.float64))
    new = batch.behavior_log_probs.clone().requires_grad_()
    ppo_actor_loss(new, batch).loss.backward()
    assert new.grad is not None
    assert old.grad is None and rewards.grad is None and values.grad is None
    assert not batch.advantages.requires_grad


def test_homogeneous_failures_do_not_force_the_learned_baseline_advantages_to_zero():
    batch = _batch([-1.0] * 4, old_values=torch.full((4,), -0.25, dtype=torch.float64))
    torch.testing.assert_close(batch.advantages, torch.full((4,), -0.75, dtype=torch.float64))
    assert batch.advantages.mean().item() == -0.75


def test_value_snapshot_does_not_see_rewards_or_change_when_the_critic_is_fitted():
    critic = PromptValueCritic(2).double()
    ids = torch.tensor([0, 0, 1, 1])
    before = snapshot_values(critic, ids)
    batch = _batch([1.0, 0.5, -0.5, -1.0], old_values=before, prompt_ids=ids)
    fixed_advantages = batch.advantages.clone()
    opt = torch.optim.SGD(critic.parameters(), lr=DEFAULT_CRITIC_LR)
    opt.zero_grad()
    value_loss(critic(ids), batch).backward()
    opt.step()
    assert critic(ids)[0].item() > 0
    assert critic(ids)[2].item() < 0
    torch.testing.assert_close(before, torch.zeros(4, dtype=torch.float64))
    torch.testing.assert_close(batch.advantages, fixed_advantages)
    # A changed reward changes only the terminal target and its advantage.
    other = _batch([-0.1] * 4, old_values=before, prompt_ids=ids)
    torch.testing.assert_close(other.old_values, batch.old_values)


def test_critic_learns_prompt_conditioned_expected_returns_without_action_features():
    critic = PromptValueCritic(2).double()
    ids = torch.tensor([0, 0, 0, 1, 1, 1])
    batch = _batch([1.0, 0.5, 0.0, -1.0, -0.5, 0.0], prompt_ids=ids)
    opt = torch.optim.SGD(critic.parameters(), lr=DEFAULT_CRITIC_LR)
    initial_mse = value_loss(critic(ids), batch).item()
    for _ in range(100):
        opt.zero_grad()
        value_loss(critic(ids), batch).backward()
        opt.step()
    predictions = critic(torch.tensor([0, 1]))
    torch.testing.assert_close(predictions, torch.tensor([0.5, -0.5], dtype=torch.float64))
    assert value_loss(critic(ids), batch).item() < initial_mse
    # Every action sampled from the same prompt receives the same prediction.
    assert critic(ids)[:3].unique().numel() == 1
    with pytest.raises(ValueError, match="outside the training set"):
        critic(torch.tensor([2]))


def test_multiple_actor_epochs_use_frozen_behavior_and_activate_clipping():
    logits = torch.nn.Parameter(torch.tensor([0.0, 0.0], dtype=torch.float64))
    actions = torch.tensor([0, 1])
    behavior = logits.log_softmax(dim=0)[actions].detach().clone()
    batch = _batch([1.0, -1.0], old_log_probs=behavior)
    opt = torch.optim.SGD([logits], lr=0.5)
    fractions = []
    for _ in range(4):
        opt.zero_grad()
        result = ppo_actor_loss(logits.log_softmax(dim=0)[actions], batch)
        fractions.append(result.active_clip_fraction.item())
        result.loss.backward()
        opt.step()
    assert fractions[0] == 0
    assert max(fractions[1:]) > 0
    assert logits[0].item() > logits[1].item()
    torch.testing.assert_close(batch.behavior_log_probs, behavior)


def test_value_targets_are_detached_and_do_not_backpropagate_to_rewards():
    rewards = torch.tensor([0.1, 0.3], dtype=torch.float64, requires_grad=True)
    batch = _batch(rewards)
    predictions = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    loss = value_loss(predictions, batch)
    assert loss.item() == pytest.approx(0.05)
    loss.backward()
    assert rewards.grad is None
    torch.testing.assert_close(predictions.grad, torch.tensor([-0.1, -0.3], dtype=torch.float64))


def test_critic_initialization_does_not_consume_actor_sampling_rng():
    state = torch.random.get_rng_state().clone()
    critic = PromptValueCritic(6)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert critic.values.weight.count_nonzero().item() == 0


def test_minibatch_selection_preserves_frozen_advantages():
    batch = _batch([1., 2., 3.], old_values=torch.tensor([.2, .4, .6], dtype=torch.float64))
    selected = batch.select(torch.tensor([2, 0]))
    torch.testing.assert_close(selected.advantages, torch.tensor([2.4, .8], dtype=torch.float64))
    assert len(selected) == 2
    with pytest.raises(ValueError, match="outside the rollout"):
        batch.select(torch.tensor([3]))


def test_input_validation_rejects_broadcasting_nonfinite_values_and_bad_clipping():
    with pytest.raises(ValueError, match="same shape"):
        _batch([1., 2.], old_values=torch.zeros(1, dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        _batch([float("nan")])
    batch = _batch([1.0])
    with pytest.raises(ValueError, match="between zero and one"):
        ppo_actor_loss(torch.zeros(1, dtype=torch.float64), batch, clip_coef=1.0)
    with pytest.raises(ValueError, match="one-dimensional"):
        ppo_actor_loss(torch.zeros(1, 1, dtype=torch.float64), batch)
    with pytest.raises(FloatingPointError, match="overflowed"):
        ppo_actor_loss(torch.tensor([1000.0], dtype=torch.float64), batch)
