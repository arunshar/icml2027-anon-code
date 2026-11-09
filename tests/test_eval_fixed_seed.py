import train_speed_grpo as T


def test_two_eval_calls_give_identical_speeds():
    args = T.build_arg_parser().parse_args(["--arm", "none", "--w", "0", "--vmax", "30", "--tiny", "--device", "cpu",
                                            "--out-dir", "/dev/null", "--no-ref", "--eval-samples", "3"])
    T.seed_everything(0)
    tok, model, _ = T.build_models(args, "cpu")
    eos = sorted({i for i in (tok.eos_token_id, tok.pad_token_id) if i is not None})
    e1 = T.evaluate(model, tok, "cpu", args, eos)
    e2 = T.evaluate(model, tok, "cpu", args, eos)
    assert e1["eval_speeds"] == e2["eval_speeds"] and e1["eval_n"] == 18
