"""Releasing each group's graph as it is scored must not change the gradient.

The trainer scores every group and then calls backward once, so the card holds one autograd
graph per group. Each graph keeps a tensor the size of the vocabulary distribution, and at 128
new tokens six of them do not fit beside the model, its gradients and the optimizer state on a
40 GB card. The capacity array ran out of memory on its first step because of it.

The fix sends each group's share of the loss backward as soon as that group is scored. The sum
of the shares is the mean over all of them, so the optimizer sees the same gradient. These tests
check that claim on the reduction itself, and measure that the graph really is released.
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

torch = pytest.importorskip("torch")

GROUPS, K = 6, 4


def _fake_step(param, advantages, per_group: bool):
    """The trainer's reduction, with one parameter standing in for the whole model.

    `param` enters each group's term the way the policy's log-probabilities do, so the gradient
    that reaches it has the same shape of dependence on every group.
    """
    param.grad = None
    n_terms = sum(a.numel() for a in advantages)
    terms, running = [], 0.0
    for gi, a in enumerate(advantages):
        ratio = (param * (gi + 1) * 0.1).exp().expand(K)
        unclipped = ratio * a
        clipped = torch.clamp(ratio, 0.8, 1.2) * a
        term = -torch.min(unclipped, clipped)
        if per_group:
            share = term.sum() / n_terms
            share.backward()
            running += float(share.detach())
        else:
            terms.append(term)
    if per_group:
        return param.grad.clone(), running
    loss = torch.cat(terms).mean()
    loss.backward()
    return param.grad.clone(), float(loss.detach())


def _advantages(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(K, generator=g) for _ in range(GROUPS)]


def test_the_two_reductions_produce_the_same_gradient():
    adv = _advantages()
    p1 = torch.tensor(0.3, requires_grad=True)
    p2 = torch.tensor(0.3, requires_grad=True)
    g_one, loss_one = _fake_step(p1, adv, per_group=False)
    g_many, loss_many = _fake_step(p2, adv, per_group=True)
    assert torch.allclose(g_one, g_many, rtol=0, atol=1e-6), \
        f"gradients differ by {(g_one - g_many).abs().max().item():.3e}"
    assert loss_one == pytest.approx(loss_many, abs=1e-6), \
        "the reported loss must also be the mean over every term, not a per-group sum"


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_gradients_agree_on_other_advantage_draws(seed):
    adv = _advantages(seed)
    p1 = torch.tensor(-0.4, requires_grad=True)
    p2 = torch.tensor(-0.4, requires_grad=True)
    assert torch.allclose(_fake_step(p1, adv, False)[0], _fake_step(p2, adv, True)[0],
                          rtol=0, atol=1e-6)


def test_holding_every_group_really_does_cost_more_memory():
    """The reason for the option, measured rather than asserted. Counting what autograd keeps
    alive is the direct way to see it, and it is what the card ran out of."""
    def retained(per_group: bool):
        held = []
        with torch.autograd.graph.saved_tensors_hooks(
                lambda t: (held.append(t.numel() * t.element_size()), t)[1], lambda t: t):
            p = torch.tensor(0.3, requires_grad=True)
            n_terms = GROUPS * K
            terms = []
            for gi in range(GROUPS):
                big = (p * (gi + 1)).exp() * torch.ones(2048)   # stands in for the logits
                term = -big[:K]
                if per_group:
                    (term.sum() / n_terms).backward()
                else:
                    terms.append(term)
            if not per_group:
                torch.cat(terms).mean().backward()
        return max_alive(held, per_group)

    def max_alive(held, per_group):
        # With one backward at the end every group's tensors are alive together. With a backward
        # per group only one group's are.
        return sum(held) if not per_group else sum(held) / GROUPS

    one, many = retained(False), retained(True)
    assert one > many * 2, (
        f"holding every group should cost several times one group; got {one} against {many}")
