import pytest

from app.numerals import expand_date_numerals


@pytest.mark.parametrize(
    "text, expected",
    [
        ("16日まで", "じゅうろくにちまで"),
        ("11日は", "じゅういちにちは"),
        ("10月ぐらい", "じゅうがつぐらい"),
        # Irregular day readings, the ones a naive digit-by-digit reading
        # would get wrong (1-10, 14, 20, 24).
        ("1日", "ついたち"),
        ("2日", "ふつか"),
        ("14日", "じゅうよっか"),
        ("20日", "はつか"),
        ("24日", "にじゅうよっか"),
        # Irregular month readings (4,7,9).
        ("4月", "しがつ"),
        ("7月", "しちがつ"),
        ("9月", "くがつ"),
    ],
)
def test_expand_date_numerals(text, expected):
    assert expand_date_numerals(text) == expected


def test_leaves_out_of_range_numbers_alone():
    # No 32nd day / 13th month — nothing to substitute, so leave as-is
    # rather than guess.
    assert expand_date_numerals("32日") == "32日"
    assert expand_date_numerals("13月") == "13月"


def test_leaves_unsupported_counters_alone():
    # 番/年 are open-ended constructions with no closed-vocabulary
    # dictionary entry to substitute in — see module docstring.
    assert expand_date_numerals("8番テーブルに2024年") == "8番テーブルに2024年"


def test_leaves_text_without_dates_alone():
    text = "気象庁によると、とても強く降る所がありそうです。"
    assert expand_date_numerals(text) == text


def test_multiple_dates_in_one_sentence():
    assert expand_date_numerals("16日から20日まで") == "じゅうろくにちからはつかまで"
