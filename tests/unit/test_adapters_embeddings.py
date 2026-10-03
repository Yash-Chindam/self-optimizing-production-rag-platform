import math
import sys
import types
from typing import Any

from rag_platform.adapters.embeddings import HashingEmbedder, SentenceTransformerEmbedder


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def test_every_vector_has_the_configured_dimension() -> None:
    embedder = HashingEmbedder(dimensions=128)
    assert embedder.dimensions == 128
    assert all(len(vector) == 128 for vector in embedder.embed(["one", "two"]))


def test_vectors_are_l2_normalized_so_cosine_is_a_dot_product() -> None:
    [vector] = HashingEmbedder().embed(["Annual leave requests use the HR portal."])
    assert math.isclose(math.sqrt(sum(value * value for value in vector)), 1.0, abs_tol=1e-9)


def test_embedding_is_deterministic_across_calls_and_instances() -> None:
    text = "Annual leave requests use the HR portal."
    assert HashingEmbedder().embed([text]) == HashingEmbedder().embed([text])


def test_related_text_scores_higher_than_unrelated_text() -> None:
    embedder = HashingEmbedder()
    [query, related, unrelated] = embedder.embed(
        [
            "annual leave request",
            "Annual leave requests use the HR portal.",
            "The server rack was replaced in the Dublin data centre.",
        ]
    )
    assert cosine(query, related) > cosine(query, unrelated)


def test_a_synonym_lands_near_the_indexed_vocabulary() -> None:
    """The embedder tokenizes semantically, so `vacation` and `leave` share features."""
    embedder = HashingEmbedder()
    [vacation, leave, unrelated] = embedder.embed(
        ["vacation request", "leave request", "server rack replacement"]
    )
    assert cosine(vacation, leave) > cosine(vacation, unrelated)


def test_empty_text_produces_a_zero_vector_rather_than_failing() -> None:
    [vector] = HashingEmbedder(dimensions=32).embed([""])
    assert vector == [0.0] * 32


def test_bigrams_can_be_turned_off() -> None:
    text = "annual leave request"
    with_bigrams = HashingEmbedder(use_bigrams=True).embed([text])
    without = HashingEmbedder(use_bigrams=False).embed([text])
    assert with_bigrams != without


def test_the_revision_is_reported_because_an_index_version_records_it() -> None:
    assert HashingEmbedder().revision == "embed-hashing-v1"


def test_embedding_a_batch_matches_embedding_one_at_a_time() -> None:
    embedder = HashingEmbedder()
    batch = embedder.embed(["alpha text", "beta text"])
    assert batch == [embedder.embed(["alpha text"])[0], embedder.embed(["beta text"])[0]]


class FakeSentenceTransformer:
    """Stands in for the real model so the lazy-load path is covered without a download."""

    instances = 0

    def __init__(self, model_name: str) -> None:
        FakeSentenceTransformer.instances += 1
        self.model_name = model_name

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def encode(
        self, texts: list[str], normalize_embeddings: bool = False, convert_to_numpy: bool = False
    ) -> list[list[float]]:
        return [[float(len(text)), 0.0, 1.0] for text in texts]


def install_fake_sentence_transformers(monkeypatch: Any) -> None:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    FakeSentenceTransformer.instances = 0


def test_the_learned_embedder_reports_the_models_dimension(monkeypatch: Any) -> None:
    install_fake_sentence_transformers(monkeypatch)
    assert SentenceTransformerEmbedder().dimensions == 3


def test_the_learned_embedder_passes_text_through_to_the_model(monkeypatch: Any) -> None:
    install_fake_sentence_transformers(monkeypatch)
    assert SentenceTransformerEmbedder().embed(["abcd"]) == [[4.0, 0.0, 1.0]]


def test_the_model_is_loaded_once_and_reused(monkeypatch: Any) -> None:
    """The model is the expensive part, so it must not be constructed per call."""
    install_fake_sentence_transformers(monkeypatch)
    embedder = SentenceTransformerEmbedder()
    embedder.embed(["one"])
    embedder.embed(["two"])
    assert embedder.dimensions == 3
    assert FakeSentenceTransformer.instances == 1


def test_the_model_is_not_loaded_until_it_is_needed(monkeypatch: Any) -> None:
    install_fake_sentence_transformers(monkeypatch)
    SentenceTransformerEmbedder()
    assert FakeSentenceTransformer.instances == 0
