"""文本切分器：面向中英文的递归字符切分。"""
from __future__ import annotations

from stockg.domain import Chunk, Document, TextSplitter


class RecursiveCharacterSplitter(TextSplitter):
    """按分隔符优先级递归切分，尽量保留语义完整（句子 / 段落）。

    中文没有天然空格，因此把句号、问号、逗号等标点也作为切分点。
    """

    def __init__(self) -> None:
        # 优先级从高到低：段落 -> 换行 -> 句末标点 -> 逗号 -> 空格 -> 字符
        self._separators = ["\n\n", "\n", "。", "！", "？", "；", "，", ".", " ", ""]

    def split(
        self, doc: Document, chunk_size: int = 400, chunk_overlap: int = 50
    ) -> list[Chunk]:
        pieces = self._split_text(doc.text, chunk_size, chunk_overlap)
        pieces = [p.strip() for p in pieces if p.strip()]
        return [Chunk(text=p, source=doc.source) for p in pieces]

    def _split_text(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        if chunk_overlap >= chunk_size:
            chunk_overlap = 0

        def _recursive_split(part: str, seps: list[str]) -> list[str]:
            if len(part) <= chunk_size:
                return [part]
            sep = seps[0] if seps else ""
            if sep == "":
                # 按字符硬切
                return [part[i : i + chunk_size] for i in range(0, len(part), chunk_size)]
            if sep not in part:
                return _recursive_split(part, seps[1:])
            segments = part.split(sep)
            merged: list[str] = []
            current = ""
            for index, seg in enumerate(segments):
                suffix = sep if index < len(segments) - 1 else ""
                candidate = current + seg + suffix if current else seg + suffix
                if len(candidate) > chunk_size and current:
                    merged.extend(_recursive_split(current, seps[1:]))
                    current = seg + suffix
                else:
                    current = candidate
            if current:
                merged.extend(_recursive_split(current, seps[1:]))
            return merged

        raw = _recursive_split(text, self._separators)
        # 合并过小的片段，并施加 overlap 滑动窗口
        merged: list[str] = []
        buf = ""
        for piece in raw:
            if len(buf) + len(piece) <= chunk_size:
                buf += piece
            else:
                if buf:
                    merged.append(buf)
                buf = piece
        if buf:
            merged.append(buf)

        if chunk_overlap <= 0 or len(merged) <= 1:
            return merged

        windowed: list[str] = []
        for i, piece in enumerate(merged):
            windowed.append(piece)
            if i < len(merged) - 1:
                nxt = merged[i + 1]
                if len(piece) < chunk_size:
                    extra = nxt[: chunk_overlap]
                    windowed[-1] = piece + extra
        return windowed
