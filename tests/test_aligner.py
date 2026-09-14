"""Unit tests for app/aligner.py's beam-retry control flow (align() itself,
not the /align endpoint — see test_align.py for that). Bypasses the real
audio-feature pipeline (Segment/KalpyUtterance/CmvnComputer all need a real
wav file and loaded MFA models) by faking just enough of it to reach
align_utterance_online, the one call these tests actually exercise.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
from kalpy.exceptions import AlignerError

from app import aligner, config


class _FakeUtterance:
    def __init__(self, segment, transcript):
        self.segment = segment
        self.transcript = transcript
        self.mfccs = object()

    def generate_mfccs(self, _mfcc_computer):
        pass

    def apply_cmvn(self, _cmvn):
        pass


class _FakeCmvnComputer:
    def compute_cmvn_from_features(self, _features):
        return object()


class _FakeWordInterval:
    def __init__(self, label, begin, end):
        self.label = label
        self.begin = begin
        self.end = end
        self.phones = []


class _FakeCtm:
    def __init__(self, word_intervals):
        self.word_intervals = word_intervals


@pytest.fixture(autouse=True)
def _fake_pipeline(monkeypatch):
    monkeypatch.setattr(aligner, "Segment", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(aligner, "KalpyUtterance", _FakeUtterance)
    monkeypatch.setattr(aligner, "CmvnComputer", _FakeCmvnComputer)
    monkeypatch.setattr(
        aligner,
        "_state",
        SimpleNamespace(
            acoustic_model=SimpleNamespace(mfcc_computer=object()),
            lexicon_compiler=object(),
            tokenizer=object(),
        ),
    )


def test_align_retries_at_a_wider_beam_after_a_default_beam_failure(monkeypatch):
    calls = []

    def fake_align_utterance_online(_acoustic_model, _utterance, _lexicon_compiler, tokenizer=None, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise AlignerError("Could not align the file with the current beam size (10)")
        return _FakeCtm([_FakeWordInterval("すき", 0.0, 0.4)])

    monkeypatch.setattr(aligner, "align_utterance_online", fake_align_utterance_online)

    result = aligner.align(Path("/tmp/fake.wav"), "すき")

    assert len(calls) == 2
    assert calls[0] == {}
    assert calls[1] == {
        "beam": config.ALIGN_FAILURE_RETRY_BEAM,
        "retry_beam": config.ALIGN_FAILURE_RETRY_BEAM * 4,
    }
    assert result["words"][0]["text"] == "すき"


def test_align_does_not_retry_when_the_default_beam_succeeds(monkeypatch):
    calls = []

    def fake_align_utterance_online(_acoustic_model, _utterance, _lexicon_compiler, tokenizer=None, **kwargs):
        calls.append(kwargs)
        return _FakeCtm([_FakeWordInterval("すき", 0.0, 0.4)])

    monkeypatch.setattr(aligner, "align_utterance_online", fake_align_utterance_online)

    aligner.align(Path("/tmp/fake.wav"), "すき")

    assert len(calls) == 1


def test_align_propagates_when_even_the_wider_beam_fails(monkeypatch):
    def always_fails(*_args, **_kwargs):
        raise AlignerError("still could not align")

    monkeypatch.setattr(aligner, "align_utterance_online", always_fails)

    with pytest.raises(AlignerError):
        aligner.align(Path("/tmp/fake.wav"), "すき")
