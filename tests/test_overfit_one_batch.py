"""Overfit-one-batch, loss-goes-down, and zero-advantage-means-no-update, on the tiny model."""
import numpy as np
import torch

import train_speed_grpo as T
from speed_task import PROMPTS


def _setup(seed=0):
    args = T.build_arg_parser().parse_args(["--arm", "graded", "--w", "5", "--vmax", "30", "--tiny", "--device", "cpu",
                                            "--out-dir", "/dev/null", "--no-ref", "--seed", str(seed)])
    T.seed_everything(seed)
    tok, model, _ = T.build_models(args, "cpu")
    eos = sorted({i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None})
    ids = T.chat_ids(tok, PROMPTS[0], "cpu")
    plen = ids.shape[1]
    with torch.no_grad():
        gen = model.generate(ids, do_sample=True, temperature=1.0, top_p=0.95, top_k=0, max_new_tokens=6,
                             num_return_sequences=4, pad_token_id=tok.pad_token_id)
    mask = T.completion_mask(gen, plen, eos)
    return args, model, gen, mask


def _update(model, opt, gen, mask, adv):
    """One GRPO update. At one epoch the ratio is 1 and sum(adv)=0, so the clipped surrogate
    is exactly 0 every step; the quantity the gradient moves is the advantage-weighted
    log-probability, returned as `objective`."""
    with torch.no_grad():
        logp_old = T.seq_logprobs(model, gen, mask)
    logp_new = T.seq_logprobs(model, gen, mask)
    ratio = (logp_new - logp_old).exp()
    loss = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    objective = float((adv * logp_new.detach()).mean())
    return objective, logp_new.detach()


def test_positive_advantage_sequence_logprob_rises_and_loss_falls():
    args, model, gen, mask = _setup()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    adv = torch.tensor([1.5, -0.5, -0.5, -0.5])
    objs, lp0 = [], []
    for _ in range(20):
        o, lp = _update(model, opt, gen, mask, adv)
        objs.append(o); lp0.append(float(lp[0]))
    assert lp0[-1] > lp0[0]                      # the reinforced sequence gets likelier
    assert objs[-1] > objs[0]                    # the objective the update climbs rises
    assert (np.diff(objs) >= -1e-6).mean() >= 0.9  # and does so nearly monotonically


def test_zero_advantage_group_changes_no_parameter():
    args, model, gen, mask = _setup()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    before = T.param_checksum(model)
    adv = torch.zeros(4)
    _update(model, opt, gen, mask, adv)
    assert T.param_checksum(model) == before
