"""Token estimate tests."""

from __future__ import annotations

from app.token_count import (
    anthropic_count_response,
    estimate_anthropic_input_tokens,
    estimate_responses_input_tokens,
    estimate_tokens,
    responses_input_tokens_response,
)


def test_estimate_tokens_cjk_and_latin():
    assert estimate_tokens("") == 0
    assert estimate_tokens("    ") == 0
    n_cjk = estimate_tokens("你好世界")
    n_lat = estimate_tokens("abcd" * 10)
    assert n_cjk >= 2
    assert n_lat >= 5


def test_count_endpoints_shapes():
    anth_n = estimate_anthropic_input_tokens(
        {
            "model": "m",
            "max_tokens": 10,
            "system": "sys",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )
    assert anth_n > 0
    assert anthropic_count_response(anth_n) == {"input_tokens": anth_n}

    resp_n = estimate_responses_input_tokens(
        {
            "model": "m",
            "instructions": "sys",
            "input": "hello",
        }
    )
    assert resp_n > 0
    body = responses_input_tokens_response(resp_n)
    assert body["object"] == "response.input_tokens"
    assert body["input_tokens"] == resp_n
