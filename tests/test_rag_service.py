from pathlib import Path

from stockg.application.rag_service import RagIngestionService, RagQueryService
from stockg.domain import Chunk, Document, RetrievedContext


class FakeLoader:
    def __init__(self) -> None:
        self.selected_paths: list[str] = []
        self.loaded_paths: list[str] = []

    def select(self, path: str) -> "FakeLoader":
        self.selected_paths.append(path)
        return self

    def load(self, path: str) -> Document:
        self.loaded_paths.append(path)
        return Document(source=Path(path).name, text=f"content:{Path(path).name}")


class FakeSplitter:
    def __init__(self) -> None:
        self.calls: list[tuple[Document, int, int]] = []

    def split(
        self,
        doc: Document,
        chunk_size: int = 400,
        chunk_overlap: int = 50,
    ) -> list[Chunk]:
        self.calls.append((doc, chunk_size, chunk_overlap))
        return [
            Chunk(text=f"{doc.text}:part-1", source=doc.source),
            Chunk(text=f"{doc.text}:part-2", source=doc.source),
        ]


class FakeEmbedder:
    name = "fake-embedder"
    dim = 2

    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.inputs.append(texts)
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class FakeStore:
    def __init__(self) -> None:
        self.added: list[tuple[list[Chunk], str]] = []

    def add(self, chunks: list[Chunk], model_name: str) -> None:
        self.added.append((chunks, model_name))


class FakeRetriever:
    def __init__(self, contexts: list[RetrievedContext]) -> None:
        self.contexts = contexts
        self.calls: list[tuple[str, int]] = []

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedContext]:
        self.calls.append((query, top_k))
        return self.contexts


def _build_service() -> tuple[
    RagIngestionService,
    FakeLoader,
    FakeSplitter,
    FakeEmbedder,
    FakeStore,
]:
    loader = FakeLoader()
    splitter = FakeSplitter()
    embedder = FakeEmbedder()
    store = FakeStore()
    service = RagIngestionService(loader.select, splitter, embedder, store)
    return service, loader, splitter, embedder, store


def test_ingest_file_runs_pipeline_and_attaches_embeddings(tmp_path) -> None:
    service, loader, splitter, embedder, store = _build_service()
    path = tmp_path / "report.md"
    path.write_text("ignored by fake", encoding="utf-8")

    count = service.ingest_file(str(path), chunk_size=120, chunk_overlap=12)

    assert count == 2
    assert loader.loaded_paths == [str(path)]
    document, chunk_size, chunk_overlap = splitter.calls[0]
    assert document.source == "report.md"
    assert (chunk_size, chunk_overlap) == (120, 12)
    assert embedder.inputs == [["content:report.md:part-1", "content:report.md:part-2"]]
    chunks, model_name = store.added[0]
    assert model_name == "fake-embedder"
    assert [chunk.embedding for chunk in chunks] == [[0.0, 1.0], [1.0, 1.0]]


def test_ingest_path_recurses_supported_files_in_sorted_order(tmp_path) -> None:
    service, loader, _, _, store = _build_service()
    (tmp_path / "nested").mkdir()
    for relative_path in ["b.txt", "a.md", "nested/c.pdf", "ignored.csv"]:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content", encoding="utf-8")

    count = service.ingest_path(str(tmp_path), chunk_size=80, chunk_overlap=8)

    assert count == 6
    expected_names = ["a.md", "b.txt", "c.pdf"]
    assert [Path(path).name for path in loader.selected_paths] == expected_names
    assert [Path(path).name for path in loader.loaded_paths] == expected_names
    assert len(store.added) == 3


def test_ingest_path_delegates_non_directory_to_ingest_file(tmp_path) -> None:
    service, loader, _, _, _ = _build_service()
    path = tmp_path / "report.md"

    count = service.ingest_path(str(path))

    assert count == 2
    assert loader.loaded_paths == [str(path)]


def test_query_service_delegates_question_and_top_k() -> None:
    contexts = [RetrievedContext("text", "source", 0.9)]
    retriever = FakeRetriever(contexts)
    service = RagQueryService(retriever)

    result = service.retrieve("主要风险是什么？", top_k=7)

    assert result is contexts
    assert retriever.calls == [("主要风险是什么？", 7)]
