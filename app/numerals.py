"""Expands arabic-digit date expressions (16日, 10月) to their hiragana
reading (じゅうろくにち, じゅうがつ) before alignment.

Root cause (jp_sentence_splits docs/STATUS.md 2026-09-18, "word-clip
boundary" backfill investigation): mined transcripts render numbers as
arabic digits (Whisper's convention), but this service's MFA dictionary
has no lexicon entry for a bare digit string — "16" alone has no
pronunciation, so the tokenizer emits `<unk>` for the whole "16日" token.
`isolatedWordRange` (jp_sentence_splits `src/lib/isolatedWordRange.ts`)
treats any `<unk>` as poisoning every later word in the same sentence (the
`<unk>` consumes real audio duration but contributes no characters to its
character-proportion basis, so every word after it drifts) — one date at
the start of a weather-report sentence was enough to knock out isolation
for every word after it.

Two things ruled out first, both checked directly against the dictionary
file (`japanese_mfa.dict`):
  - Converting the digits to kanji (16日 -> 十六日) isn't enough on its
    own: 十六日/十一日 have no entry of their own (only a handful of very
    common ones like 十月 do — the tokenizer emits the rest as a single
    indivisible token it can't look up either way).
  - Inserting an explicit space to force the tokenizer to split the
    number from the counter doesn't work — it broke the *already-working*
    cases too, so whatever this tokenizer does with a literal space
    character, it isn't "treat it as a word boundary."

What does work: day-of-month and month-name readings are closed,
frequently-occurring vocabulary, and the dictionary FILE has a literal
whole-compound hiragana entry for every single one (1-31日 including the
irregulars, 1-12月). But a live end-to-end sweep against `/align` (all 43
values, docs/STATUS.md 2026-09-18) showed the dictionary entry alone
doesn't guarantee success: the tokenizer routinely splits the substituted
reading into smaller pieces *before* lexicon lookup (e.g. じゅうろくにち
-> "じゅう" + "ろくにち") regardless of the whole string being a real
dictionary entry, and only succeeds when *every* resulting piece
independently happens to also be real vocabulary.

Getting the tokenizer to keep the compound whole isn't achievable from
input text alone (see git history — a forced space boundary was tried and
regressed already-working cases). What *is* achievable: querying the same
spaCy tokenizer directly (`generate_language_tokenizer`) shows exactly
which fragment it produces for each failing value, and every failing
tens-prefixed day (13, 16-19, 23, 26-30) reduces to one of six recurring
"ones-digit + にち" remainders — さんにち/ろくにち/しちにち/はちにち/
くにち/じゅうにち — none of which are independent vocabulary. Their
correct phones aren't a guess: each is the verified whole-compound's
phone sequence (already a real dictionary entry) with the verified
prefix's phones removed from the front, cross-checked two ways (the same
remainder falls out whether you subtract from the じゅう-compound or the
にじゅう-compound) — added to `app/data/supplementary_dictionary.dict`
once confirmed. Result: **12/12 months, 30/31 days**. The one holdout,
day 3 (みっか -> "みっ" + "か"), is different in kind: the tokenizer's
split falls *inside* a sokuon gemination (the っ's length is realized on
the following consonant, not attributable to either resulting token
cleanly), so there's no confident way to assign phones to "みっ" the way
there was for the others — deliberately left alone rather than guess at a
phonetic entry with no verified data to derive it from. A day that still
fails behaves exactly as before this fix (still `<unk>`, same cascade) —
never worse, just not fixed.

Other counters (番, 年) are NOT covered: 8番/2024年 are open-ended
constructions, not closed vocabulary — grep found no compound entry for
any value tried, kana or kanji, so there is nothing this function can
substitute that would actually resolve, and no verified data to derive
missing-fragment phones from the way there was for 日/月.
"""
from __future__ import annotations

import re

_DAY_READINGS: dict[int, str] = {
    1: "ついたち",
    2: "ふつか",
    3: "みっか",
    4: "よっか",
    5: "いつか",
    6: "むいか",
    7: "なのか",
    8: "ようか",
    9: "ここのか",
    10: "とおか",
    11: "じゅういちにち",
    12: "じゅうににち",
    13: "じゅうさんにち",
    14: "じゅうよっか",
    15: "じゅうごにち",
    16: "じゅうろくにち",
    17: "じゅうしちにち",
    18: "じゅうはちにち",
    19: "じゅうくにち",
    20: "はつか",
    21: "にじゅういちにち",
    22: "にじゅうににち",
    23: "にじゅうさんにち",
    24: "にじゅうよっか",
    25: "にじゅうごにち",
    26: "にじゅうろくにち",
    27: "にじゅうしちにち",
    28: "にじゅうはちにち",
    29: "にじゅうくにち",
    30: "さんじゅうにち",
    31: "さんじゅういちにち",
}

_MONTH_READINGS: dict[int, str] = {
    1: "いちがつ",
    2: "にがつ",
    3: "さんがつ",
    4: "しがつ",
    5: "ごがつ",
    6: "ろくがつ",
    7: "しちがつ",
    8: "はちがつ",
    9: "くがつ",
    10: "じゅうがつ",
    11: "じゅういちがつ",
    12: "じゅうにがつ",
}

_DATE_PATTERN = re.compile(r"(\d+)(日|月)")


def expand_date_numerals(text: str) -> str:
    """Replaces `<n>日`/`<n>月` with its hiragana reading when `<n>` is a
    valid day-of-month (1-31) or month number (1-12); leaves anything else
    (other counters, out-of-range numbers) untouched."""

    def repl(match: re.Match[str]) -> str:
        n = int(match.group(1))
        table = _DAY_READINGS if match.group(2) == "日" else _MONTH_READINGS
        reading = table.get(n)
        return reading if reading is not None else match.group(0)

    return _DATE_PATTERN.sub(repl, text)
