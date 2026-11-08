import json
from pathlib import Path

import pytest

from app.number_expansion import expand_numerals, number_reading

CASES = json.loads((Path(__file__).parent / "fixtures" / "numeral_expansion_cases.json").read_text(encoding="utf8"))


@pytest.mark.parametrize("case", CASES, ids=[c["input"] for c in CASES])
def test_shared_fixture(case):
    # The same fixture is checked against src/lib/alignerText.ts in jp_sentence_splits.
    assert expand_numerals(case["input"]) == case["expected"]


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, "ぜろ"),
        (10, "じゅう"),
        (300, "さんびゃく"),
        (600, "ろっぴゃく"),
        (800, "はっぴゃく"),
        (3000, "さんぜん"),
        (8000, "はっせん"),
        (10000, "いちまん"),
        (12345, "いちまんにせんさんびゃくよんじゅうご"),
    ],
)
def test_number_reading(n, expected):
    assert number_reading(n) == expected


def test_dates_still_use_date_readings():
    assert expand_numerals("1日と20日") == "ついたちとはつか"


def test_text_without_digits_untouched():
    text = "気象庁によると、強く降る所があります。"
    assert expand_numerals(text) == text
