"""Text embedding behind one protocol (specification section 11, "Embedding model").

The embedding revision is part of an `IndexVersion`'s fingerprint, so changing embedder is a new
index version and a new candidate configuration — never an in-place mutation of a live index.

Two implementations ship. `HashingEmbedder` is a real vector embedding (feature hashing over
token unigrams and bigrams, L2-normalized) that needs no model download, so the default install
can populate and query a vector store offline; it captures lexical and co-occurrence similarity
but not learned semantics. `SentenceTransformerEmbedder` is the learned one, installed through
the `embeddings` extra. Both satisfy `TextEmbedder`, so the index-version lifecycle, the
retrieval path and the evaluation harness do not know which is in use.
"""

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import TYPE_CHECKING, Protocol

from rag_platform.repository import tokenize

if TYPE_CHECKING:  # pragma: no cover - import used for typing only
    from sentence_transformers import SentenceTransformer


class TextEmbedder(Protocol):
    @property
    def revision(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(slots=True)
class HashingEmbedder:
    """Feature hashing over unigrams and bigrams, L2-normalized for cosine distance."""

    dimensions: int = 256
    revision: str = "embed-hashing-v1"
    use_bigrams: bool = True

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for feature in self._features(text):
            bucket, sign = self._bucket(feature)
            vector[bucket] += sign
        return _normalize(vector)

    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text, semantic=True)
        if not self.use_bigrams:
            return tokens
        bigrams = [f"{left}_{right}" for left, right in pairwise(tokens)]
        return tokens + bigrams

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2s(feature.encode("utf-8"), digest_size=5).digest()
        bucket = int.from_bytes(digest[:4], "big") % self.dimensions
        # The sign bit keeps unrelated features from only ever adding to each other.
        return bucket, 1.0 if digest[4] & 1 else -1.0


@dataclass(slots=True)
class SentenceTransformerEmbedder:
    """Learned embeddings. Requires the `embeddings` extra."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    revision: str = "embed-minilm-l6-v2"
    _model: "SentenceTransformer | None" = field(default=None, init=False, repr=False)
    _dimensions: int = field(default=0, init=False, repr=False)

    @property
    def dimensions(self) -> int:
        if not self._dimensions:
            self._dimensions = int(self._loaded().get_sentence_embedding_dimension() or 0)
        return self._dimensions

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._loaded().encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return [[float(value) for value in vector] for vector in vectors]

    def _loaded(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
