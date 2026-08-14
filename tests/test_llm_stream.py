import asyncio
from types import SimpleNamespace

from stockg.infrastructure.llm_stream import stream_completion, stream_completion_async


def _chunk(
    *,
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str | None = None,
    usage=None,
    has_choice: bool = True,
):
    choice = SimpleNamespace(
        delta=SimpleNamespace(content=content, tool_calls=tool_calls),
        finish_reason=finish_reason,
    )
    return SimpleNamespace(usage=usage, choices=[choice] if has_choice else [])


def _tool_delta(
    index: int, call_id: str | None, name: str | None, arguments: str | None
):
    function = SimpleNamespace(name=name, arguments=arguments)
    return SimpleNamespace(index=index, id=call_id, function=function)


class FakeCompletions:
    def __init__(self, chunks: list) -> None:
        self.chunks = chunks
        self.kwargs: dict | None = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return iter(self.chunks)


class FakeAsyncStream:
    def __init__(self, chunks: list) -> None:
        self.chunks = chunks

    def __aiter__(self):
        async def iterate():
            for chunk in self.chunks:
                yield chunk

        return iterate()


class FakeAsyncCompletions:
    def __init__(self, chunks: list) -> None:
        self.chunks = chunks
        self.kwargs: dict | None = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return FakeAsyncStream(self.chunks)


def test_stream_completion_reassembles_text_tools_and_usage() -> None:
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    chunks = [
        _chunk(content="分析"),
        _chunk(
            tool_calls=[
                _tool_delta(0, "call-1", "submit_final_report", '{"investment_')
            ]
        ),
        _chunk(
            tool_calls=[_tool_delta(0, None, None, 'rating":"观望"}')],
            finish_reason="tool_calls",
        ),
        _chunk(usage=usage, has_choice=False),
    ]
    completions = FakeCompletions(chunks)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    streamed_text: list[str] = []

    message, actual_usage, finish_reason = stream_completion(
        client,
        model="deepseek-chat",
        messages=[{"role": "user", "content": "question"}],
        tools=[{"type": "function"}],
        tool_choice="auto",
        max_tokens=256,
        on_text=streamed_text.append,
    )

    assert message == {
        "role": "assistant",
        "content": "分析",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "submit_final_report",
                    "arguments": '{"investment_rating":"观望"}',
                },
            }
        ],
    }
    assert actual_usage is usage
    assert finish_reason == "tool_calls"
    assert streamed_text == ["分析"]
    assert completions.kwargs["stream"] is True
    assert completions.kwargs["stream_options"] == {"include_usage": True}


def test_stream_completion_returns_none_for_empty_content_and_tools() -> None:
    completions = FakeCompletions([_chunk(finish_reason="stop")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    message, usage, finish_reason = stream_completion(
        client,
        model="model",
        messages=[],
        tools=[],
        tool_choice="auto",
        max_tokens=32,
    )

    assert message == {"role": "assistant", "content": None, "tool_calls": None}
    assert usage is None
    assert finish_reason == "stop"


def test_stream_completion_async_awaits_async_text_callback() -> None:
    chunks = [_chunk(content="第一段"), _chunk(content="第二段", finish_reason="stop")]
    completions = FakeAsyncCompletions(chunks)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    received: list[str] = []

    async def on_text(text: str) -> None:
        await asyncio.sleep(0)
        received.append(text)

    async def scenario():
        return await stream_completion_async(
            client,
            model="model",
            messages=[],
            tools=[],
            tool_choice="auto",
            max_tokens=32,
            on_text=on_text,
        )

    message, usage, finish_reason = asyncio.run(scenario())

    assert message["content"] == "第一段第二段"
    assert message["tool_calls"] is None
    assert usage is None
    assert finish_reason == "stop"
    assert received == ["第一段", "第二段"]
