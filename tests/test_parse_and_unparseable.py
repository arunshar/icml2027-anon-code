from speed_task import PROMPTS, parse_speed


def test_parse_first_integer():
    assert parse_speed("25 knots") == 25.0
    assert parse_speed("-3") == -3.0
    assert parse_speed("Set it to 12, maybe 15") == 12.0


def test_parse_none_on_no_integer():
    assert parse_speed("none") is None
    assert parse_speed("") is None


def test_six_prompts_unchanged():
    assert len(PROMPTS) == 6
    assert all("knots" in p for p in PROMPTS)
