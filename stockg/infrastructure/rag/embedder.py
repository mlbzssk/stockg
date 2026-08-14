"""向量化(Embedding)实现：本地 BGE-zh，缺依赖时回退轻量哈希 Embedder。"""
from __future__ import annotations

import hashlib
import logging

import numpy as np

from stockg.domain import Embedder

logger = logging.getLogger(__name__)

# BGE 官方推荐的查询前缀：检索时给 query 加这个前缀能显著提升召回
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class BgeZhEmbedder(Embedder):
    """基于 sentence-transformers 的本地 BGE-zh 模型（推荐）。"""

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5") -> None:
        self._model_name = model_name
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # noqa: BLE001
            raise RuntimeError(
                "未安装 sentence-transformers，无法使用本地 BGE 向量化。"
                "请先: pip install sentence-transformers"
            ) from e
        self._model = SentenceTransformer(model_name)

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32).tolist()


class FallbackEmbedder(Embedder):
    """无需任何 ML 依赖的轻量哈希 Embedder。

    仅做字符 + 二元文法的哈希词袋并做 L2 归一化，语义能力很弱，
    仅用于在没有 torch / sentence-transformers 时让 RAG 流程跑通。
    装好 sentence-transformers 后会自动切换到真正的 BGE-zh。
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return f"fallback-hash-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        # 字符 + 二元文法，覆盖中英文
        tokens: list[str] = list(text)
        tokens += [text[i : i + 2] for i in range(len(text) - 1)]
        for tok in tokens:
            digest = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self._dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


def build_embedder(prefer: str = "bge") -> Embedder:
    """构造 Embedder：优先本地 BGE-zh，不可用时回退哈希 Embedder。"""
    if prefer == "bge":
        try:
            return BgeZhEmbedder()
        except Exception as e:  # noqa: BLE001 - 优雅降级
            logger.warning(
                "本地 BGE 模型不可用，回退到轻量哈希 Embedder；检索质量会降低: %s",
                e,
            )
            return FallbackEmbedder()
    return FallbackEmbedder()
