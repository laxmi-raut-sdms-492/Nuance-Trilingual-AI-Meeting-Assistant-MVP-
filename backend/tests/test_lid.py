"""
Cheap spoken language ID — the posterior wrapper, not the model.

The VoxLingua107 weights are a 100+ language classifier downloaded on first
use; nothing here loads them. What is tested is the code around them, which is
where the pipeline's assumptions live: that a 107-way guess is cut down to our
three and renormalized, that an uninformative window returns something usable
instead of raising, and that a model which will not load fails once and stays
failed rather than being retried per segment.
"""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from config import ALLOWED_LANGUAGES
import models.lid as lid


@pytest.fixture(autouse=True)
def _reset_module_state():
    """
    lid caches the classifier and any load error in module globals for the
    life of the process. Tests must not inherit each other's cache.
    """
    saved = (lid._classifier, lid._code_to_index, lid._load_error)
    lid._classifier, lid._code_to_index, lid._load_error = None, None, None
    yield
    lid._classifier, lid._code_to_index, lid._load_error = saved


class _FakeEncoder:
    def __init__(self, ind2lab):
        self.ind2lab = ind2lab


class _FakeHparams:
    def __init__(self, ind2lab):
        self.label_encoder = _FakeEncoder(ind2lab)


class _FakeClassifier:
    """Stands in for EncoderClassifier: log-probs over a toy label set."""

    def __init__(self, probs: list[float], ind2lab: dict | None = None):
        self._probs = probs
        self.hparams = _FakeHparams(
            ind2lab if ind2lab is not None else {0: "en: English", 1: "hi: Hindi", 2: "mr: Marathi"}
        )

    def classify_batch(self, _tensor):
        out_prob = torch.log(torch.tensor([self._probs], dtype=torch.float32))
        return out_prob, None, None, None


def _install(classifier) -> None:
    """Pretend the model loaded, skipping the download path entirely."""
    lid._classifier = classifier
    lid._code_to_index = lid._build_code_index(classifier)


def _audio(seconds: float = 1.0) -> np.ndarray:
    return np.full(int(seconds * 16000), 0.05, dtype=np.float32)


@contextmanager
def _fake_speechbrain(classifier):
    """
    Stand in for the speechbrain package itself.

    lid imports EncoderClassifier inside the load, so the seam is sys.modules
    rather than an attribute patch — and going through sys.modules means these
    tests behave the same whether or not speechbrain is installed.
    """
    loader = MagicMock(return_value=classifier)
    classifiers = types.ModuleType("speechbrain.inference.classifiers")
    classifiers.EncoderClassifier = types.SimpleNamespace(from_hparams=loader)

    inference = types.ModuleType("speechbrain.inference")
    inference.classifiers = classifiers
    package = types.ModuleType("speechbrain")
    package.inference = inference

    with patch.dict(
        sys.modules,
        {
            "speechbrain": package,
            "speechbrain.inference": inference,
            "speechbrain.inference.classifiers": classifiers,
        },
    ):
        yield loader


# ------------------------------------------------------------- the label map


def test_code_index_parses_voxlingua_labels():
    """VoxLingua107 labels read 'en: English' — the code is before the colon."""
    index = lid._build_code_index(_FakeClassifier([1.0, 0.0, 0.0]))
    assert index == {"en": 0, "hi": 1, "mr": 2}


def test_code_index_ignores_languages_we_do_not_transcribe():
    classifier = _FakeClassifier(
        [0.0] * 4,
        ind2lab={0: "fr: French", 1: "en: English", 2: "de: German", 3: "mr: Marathi"},
    )
    assert lid._build_code_index(classifier) == {"en": 1, "mr": 3}


def test_code_index_is_empty_when_the_model_exposes_no_labels():
    """A model with no label encoder is the wrong model, not a working one."""

    class _Bare:
        pass

    assert lid._build_code_index(_Bare()) == {}


# --------------------------------------------------------------- posteriors


def test_posterior_is_restricted_and_renormalized():
    """
    Most of a 107-way model's mass sits on languages this meeting cannot
    contain. What matters is the split among ours, so the rest is discarded
    and the remainder rescaled to sum to one.
    """
    classifier = _FakeClassifier(
        [0.10, 0.05, 0.05, 0.80],
        ind2lab={0: "en: English", 1: "hi: Hindi", 2: "mr: Marathi", 3: "ja: Japanese"},
    )
    _install(classifier)

    posterior = lid.language_posterior(_audio())

    assert set(posterior) == {"en", "hi", "mr"}
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert posterior["en"] == pytest.approx(0.5)
    assert posterior["hi"] == pytest.approx(0.25)
    assert posterior["mr"] == pytest.approx(0.25)


def test_empty_audio_returns_a_flat_distribution():
    """
    An empty window is a normal event in a boundary search, not an error.
    Flat means 'no opinion', which lcd.py rejects on confidence.
    """
    _install(_FakeClassifier([1.0, 0.0, 0.0]))

    for audio in (None, np.array([], dtype=np.float32)):
        posterior = lid.language_posterior(audio)
        assert sum(posterior.values()) == pytest.approx(1.0)
        assert len(set(posterior.values())) == 1


def test_all_mass_outside_our_languages_returns_flat_not_a_divide_by_zero():
    classifier = _FakeClassifier(
        [1.0, 0.0],
        ind2lab={0: "ja: Japanese", 1: "en: English"},
    )
    _install(classifier)

    posterior = lid.language_posterior(_audio())
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert len(set(posterior.values())) == 1


def test_labels_beyond_the_output_are_skipped_not_indexed():
    """A label map longer than the output layer must not raise IndexError."""
    classifier = _FakeClassifier([0.6, 0.4])
    classifier.hparams.label_encoder.ind2lab = {
        0: "en: English",
        1: "hi: Hindi",
        9: "mr: Marathi",
    }
    _install(classifier)

    posterior = lid.language_posterior(_audio())
    assert set(posterior) == {"en", "hi"}
    assert sum(posterior.values()) == pytest.approx(1.0)


def test_top_language_returns_the_argmax_and_its_probability():
    _install(_FakeClassifier([0.2, 0.7, 0.1]))

    code, prob = lid.top_language(_audio())
    assert code == "hi"
    assert prob == pytest.approx(0.7)


# ------------------------------------------------------------- availability


def test_is_available_is_false_and_silent_when_the_model_cannot_load():
    """
    lcd.py calls this per segment to decide whether a split is possible. It
    must answer, never throw — an unavailable model degrades to unsplit
    segments, which is the behaviour that shipped before this existed.
    """
    with patch.object(lid, "_get_classifier", side_effect=RuntimeError("no weights")):
        assert lid.is_available() is False


def test_is_available_is_true_once_the_model_is_loaded():
    _install(_FakeClassifier([1.0, 0.0, 0.0]))
    assert lid.is_available() is True


def test_a_load_failure_is_permanent_and_not_retried_per_window():
    """
    Downloading weights that will not download is not fixed by trying again,
    and retrying once per window turns a clear failure into a very slow one.
    A recorded failure must short-circuit before the loader is reached.
    """
    lid._load_error = RuntimeError("weights missing")

    with _fake_speechbrain(_FakeClassifier([1.0, 0.0, 0.0])) as loader:
        assert lid.is_available() is False
        assert lid.is_available() is False
        loader.assert_not_called()


def test_a_model_without_any_of_our_languages_is_rejected_as_wrong():
    """
    A classifier that cannot express en/hi/mr would return a flat posterior
    for every window — silently disabling the boundary search rather than
    reporting that the wrong model was configured.
    """
    classifier = _FakeClassifier([0.5, 0.5], ind2lab={0: "ja: Japanese", 1: "de: German"})

    with _fake_speechbrain(classifier):
        with pytest.raises(RuntimeError, match="wrong model"):
            lid._get_classifier()

    assert lid._load_error is not None


def test_a_missing_speechbrain_is_recorded_as_a_permanent_failure():
    """
    The import lives inside the load's try block precisely so that an absent
    dependency is cached like any other load error. Python does not cache
    failed imports, so otherwise every segment would re-walk the import path.
    """
    absent = dict.fromkeys(
        ["speechbrain", "speechbrain.inference", "speechbrain.inference.classifiers"]
    )
    with patch.dict(sys.modules, absent):
        assert lid.is_available() is False

    assert isinstance(lid._load_error, ImportError)


def test_uniform_covers_exactly_the_allowed_languages():
    assert set(lid._uniform()) == set(ALLOWED_LANGUAGES)
    assert sum(lid._uniform().values()) == pytest.approx(1.0)
