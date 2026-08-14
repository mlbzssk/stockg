import math

import pytest

from stockg.domain import Document, RetrievedContext
from stockg.infrastructure.rag import embedder as embedder_module
from stockg.infrastructure.rag.chunker import RecursiveCharacterSplitter
from stockg.infrastructure.rag.embedder import FallbackEmbedder, build_embedder
from stockg.infrastructure.rag.pdf_loader import PyPdfLoader, TextFileLoader, build_loader
from stockg.infrastructure.rag.retriever import SimpleRetriever


class RecordingEmbedder:
    name = "recording"
    dim = 2

    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.inputs.append(texts)
        return [[0.25, 0.75] for _ in texts]


class RecordingStore:
    def __init__(self) -> None:
        self.searches: list[tuple[list[float], int]] = []
        self.result = [RetrievedContext("context", "source.md", 0.8)]

    def search(self, query_vec: list[float], top_k: int) -> list[RetrievedContext]:
        self.searches.append((query_vec, top_k))
        return self.result


def test_splitter_returns_empty_list_for_blank_document() -> None:
    splitter = RecursiveCharacterSplitter()

    chunks = splitter.split(Document(source="empty.txt", text=" \n\n "), chunk_size=10)

    assert chunks == []


def test_splitter_hard_splits_long_text_and_preserves_source() -> None:
    splitter = RecursiveCharacterSplitter()
    document = Document(source="report.txt", text="abcdefghij")

    chunks = splitter.split(document, chunk_size=4, chunk_overlap=0)

    assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "ij"]
    assert {chunk.source for chunk in chunks} == {"report.txt"}
    assert all(chunk.embedding is None for chunk in chunks)


def test_splitter_disables_invalid_overlap_equal_to_chunk_size() -> None:
    splitter = RecursiveCharacterSplitter()
    document = Document(source="report.txt", text="abcdefgh")

    chunks = splitter.split(document, chunk_size=4, chunk_overlap=4)

    assert [chunk.text for chunk in chunks] == ["abcd", "efgh"]


def test_fallback_embedder_is_deterministic_and_normalized() -> None:
    embedder = FallbackEmbedder(dim=32)

    first, second, other = embedder.embed(["股票分析", "股票分析", "风险提示"])

    assert embedder.name == "fallback-hash-32"
    assert embedder.dim == 32
    assert first == second
    assert first != other
    assert math.sqrt(sum(value**2 for value in first)) == pytest.approx(1.0)


def test_fallback_embedder_returns_zero_vector_for_empty_text() -> None:
    vector = FallbackEmbedder(dim=8).embed([""])[0]

    assert vector == [0.0] * 8


def test_build_embedder_falls_back_when_bge_initialization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenBgeEmbedder:
        def __init__(self) -> None:
            raise RuntimeError("model unavailable")

    monkeypatch.setattr(embedder_module, "BgeZhEmbedder", BrokenBgeEmbedder)

    result = build_embedder("bge")

    assert isinstance(result, FallbackEmbedder)


def test_build_embedder_skips_bge_for_explicit_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called() -> None:
        raise AssertionError("BGE should not be initialized")

    monkeypatch.setattr(embedder_module, "BgeZhEmbedder", fail_if_called)

    assert isinstance(build_embedder("fallback"), FallbackEmbedder)


def test_simple_retriever_applies_prefix_and_forwards_top_k() -> None:
    embedder = RecordingEmbedder()
    store = RecordingStore()
    retriever = SimpleRetriever(embedder, store, query_prefix="query: ")

    result = retriever.retrieve("贵州茅台", top_k=5)

    assert result is store.result
    assert embedder.inputs == [["query: 贵州茅台"]]
    assert store.searches == [([0.25, 0.75], 5)]


def test_text_file_loader_reads_utf8_and_strips_outer_whitespace(tmp_path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("\n  研报正文  \n", encoding="utf-8")

    document = TextFileLoader().load(str(path))

    assert document.source == "notes.md"
    assert document.text == "研报正文"


@pytest.mark.parametrize(
    ("path", "expected_type"),
    [("REPORT.PDF", PyPdfLoader), ("notes.md", TextFileLoader), ("data.txt", TextFileLoader)],
)
def test_build_loader_selects_by_case_insensitive_suffix(path: str, expected_type: type) -> None:
    assert isinstance(build_loader(path), expected_type)
