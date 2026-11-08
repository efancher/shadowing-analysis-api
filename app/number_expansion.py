"""General arabic-digit expansion for forced alignment (bare numbers and counters).

`app.numerals.expand_date_numerals` only covers 日/月. Any other digit string
(エピソード1, 2024年, 3回, 5人) is still `<unk>` for the same reason (the MFA
dictionary has no entry for a bare digit string), and the `<unk>` consumes
real audio time without contributing characters, drifting every word around
it. `expand_numerals` reads those out as hiragana too: plain Sino-Japanese for
a bare number, and the counter's own irregular readings (sokuon/rendaku,
ひとり, よじ...) for the counters in `_COUNTERS`. Counters that read regularly
and aren't listed stay as kanji after the digits (8番 -> はち番): the
dictionary knows 番 itself.

KEEP IN SYNC with jp_sentence_splits `src/lib/alignerText.ts`; both are checked
against `tests/fixtures/numeral_expansion_cases.json` (shared verbatim).
"""
from __future__ import annotations

import re
import unicodedata

from app.numerals import expand_date_numerals

_DIGITS = ("ぜろ", "いち", "に", "さん", "よん", "ご", "ろく", "なな", "はち", "きゅう")
_MAX_NUMBER = 99_999_999


def _below_10000(n: int) -> str:
    out = ""
    thousands, rest = divmod(n, 1000)
    hundreds, rest = divmod(rest, 100)
    tens, ones = divmod(rest, 10)
    if thousands:
        out += {1: "せん", 3: "さんぜん", 8: "はっせん"}.get(thousands, _DIGITS[thousands] + "せん")
    if hundreds:
        out += {1: "ひゃく", 3: "さんびゃく", 6: "ろっぴゃく", 8: "はっぴゃく"}.get(
            hundreds, _DIGITS[hundreds] + "ひゃく"
        )
    if tens:
        out += "じゅう" if tens == 1 else _DIGITS[tens] + "じゅう"
    if ones:
        out += _DIGITS[ones]
    return out


def number_reading(n: int) -> str:
    """Plain Sino-Japanese reading of 0..99,999,999 (いち, じゅうに, さんびゃく...)."""
    if n == 0:
        return _DIGITS[0]
    man, rest = divmod(n, 10000)
    out = ""
    if man:
        out += ("いち" if man == 1 else _below_10000(man)) + "まん"
    if rest:
        out += _below_10000(rest)
    return out


# Per counter: its reading, then per last digit (0 = a round ten: 10, 20, 30...
# but not 100/1000) the (number part, counter part or None) that replaces the
# plain reading. A last digit missing from the map reads plainly. `exact` maps
# a whole number to a full reading (ひとり, ふたり, ひとつ...).
_SOKUON = {1: ("いっ", None), 6: ("ろっ", None), 8: ("はっ", None), 0: ("じゅっ", None)}


def _sokuon(extra=None, drop=()):
    table = {k: v for k, v in _SOKUON.items() if k not in drop}
    table.update(extra or {})
    return table


def _p_row(three: str, plosive: str) -> dict:
    """The は-row counters (分 ふん, 本 ほん, 杯 はい, 匹 ひき): the plosive form after 1/6/8/10, `three` after 3."""
    return {
        1: ("いっ", plosive),
        3: ("さん", three),
        6: ("ろっ", plosive),
        8: ("はっ", plosive),
        0: ("じゅっ", plosive),
    }


_COUNTERS: dict[str, dict] = {
    "回": {"reading": "かい", "digits": _sokuon()},
    "歳": {"reading": "さい", "digits": _sokuon(drop=(6,)), "exact": {20: "はたち"}},
    "点": {"reading": "てん", "digits": _sokuon()},
    "週": {"reading": "しゅう", "digits": _sokuon()},
    "個": {"reading": "こ", "digits": _sokuon()},
    "階": {"reading": "かい", "digits": _sokuon(extra={3: ("さん", "がい")})},
    "分": {"reading": "ふん", "digits": {**_p_row("ぷん", "ぷん"), 4: ("よん", "ぷん")}},
    "本": {"reading": "ほん", "digits": _p_row("ぼん", "ぽん")},
    "杯": {"reading": "はい", "digits": _p_row("ばい", "ぱい")},
    "匹": {"reading": "ひき", "digits": _p_row("びき", "ぴき")},
    "年": {"reading": "ねん", "digits": {4: ("よ", None)}},
    "時": {"reading": "じ", "digits": {4: ("よ", None), 7: ("しち", None), 9: ("く", None)}},
    "人": {"reading": "にん", "digits": {4: ("よ", None), 7: ("しち", None)}, "exact": {1: "ひとり", 2: "ふたり"}},
    "つ": {
        "reading": "つ",
        "digits": {},
        "exact": {
            1: "ひとつ",
            2: "ふたつ",
            3: "みっつ",
            4: "よっつ",
            5: "いつつ",
            6: "むっつ",
            7: "ななつ",
            8: "やっつ",
            9: "ここのつ",
            10: "とお",
        },
        "exact_only": True,
    },
}
for _alias in ("ヶ月", "ヵ月", "ケ月", "か月", "カ月"):
    _COUNTERS[_alias] = {"reading": "かげつ", "digits": _sokuon(drop=(8,))}

_COUNTER_PATTERN = "|".join(sorted((re.escape(c) for c in _COUNTERS), key=len, reverse=True))
# Not part of a decimal or comma-grouped number (3.5, 1,000).
_NUMBER_PATTERN = re.compile(rf"(?<![\d.,])(\d+)(?![.,]\d)({_COUNTER_PATTERN})?")


def _counter_reading(n: int, counter: str) -> str | None:
    spec = _COUNTERS[counter]
    exact = spec.get("exact", {})
    if n in exact:
        return exact[n]
    if spec.get("exact_only"):
        return None
    last = n % 10
    if last == 0:
        key = 0 if n % 100 != 0 else None
    else:
        key = last
    override = spec["digits"].get(key) if key is not None else None
    if override is None:
        return number_reading(n) + spec["reading"]
    num_part, counter_part = override
    counter_part = counter_part or spec["reading"]
    if key == 0:
        # 10 -> じゅっ, 20 -> にじゅっ, 110 -> ひゃくじゅっ: keep everything above the ten.
        tens_digit = (n // 10) % 10
        head = (number_reading(n - n % 100) if n >= 100 else "") + (_DIGITS[tens_digit] if tens_digit > 1 else "")
        return head + num_part + counter_part
    head = number_reading(n - last) if n > last else ""
    return head + num_part + counter_part


def expand_numerals(text: str) -> str:
    """Dates first (`expand_date_numerals`), then every remaining arabic number
    (and its counter, where irregular) rewritten to hiragana. Decimals,
    comma-grouped numbers, leading-zero strings and values past 99,999,999
    are left untouched."""
    text = expand_date_numerals(text)

    def repl(match: re.Match[str]) -> str:
        digits = unicodedata.normalize("NFKC", match.group(1))
        counter = match.group(2)
        if not digits.isdigit() or (len(digits) > 1 and digits.startswith("0")):
            return match.group(0)
        n = int(digits)
        if n > _MAX_NUMBER:
            return match.group(0)
        if counter:
            reading = _counter_reading(n, counter)
            if reading is not None:
                return reading
            return number_reading(n) + counter
        return number_reading(n)

    return _NUMBER_PATTERN.sub(repl, text)
