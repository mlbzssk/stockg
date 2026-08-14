"""文档加载器：支持 PDF 与纯文本。"""

from __future__ import annotations

from pathlib import Path

from stockg.domain import Document, DocumentLoader


class PyPdfLoader(DocumentLoader):
    """使用 pypdf 读取 PDF，抽取全部页面文本。"""

    def load(self, path: str) -> Document:
        try:
            from pypdf import PdfReader
        except ImportError as e:  # noqa: BLE001
            raise RuntimeError(
                "未安装 pypdf，无法读取 PDF。请先: pip install pypdf"
            ) from e

        reader = PdfReader(path)
        pages = [(p.extract_text() or "") for p in reader.pages]
        text = "\n".join(pages).strip()
        return Document(source=Path(path).name, text=text)


class TextFileLoader(DocumentLoader):
    """读取 .txt / .md 等纯文本文件，便于在没有 PDF 时调试。"""

    def load(self, path: str) -> Document:
        return Document(
            source=Path(path).name,
            text=Path(path).read_text(encoding="utf-8", errors="ignore").strip(),
        )


def build_loader(path: str) -> DocumentLoader:
    """根据文件后缀自动选择加载器。"""
    suffix = Path(path).suffix.lower()
    if suffix == ".pdf":
        return PyPdfLoader()
    return TextFileLoader()
