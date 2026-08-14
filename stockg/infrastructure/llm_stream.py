import asyncio
from langchain_core.tools import tool
from openai.types.chat.chat_completion import ChatCompletionMessage

def stream_completion(client, *, model, messages, tools, tool_choice, max_tokens, on_text=None):
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )

    content_parts, tool_call_parts, usage, finish_reason = [], {}, None, None
    for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason or finish_reason

        if delta.content:
            content_parts.append(delta.content)
            if on_text:
                on_text(delta.content)
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                slot = tool_call_parts.setdefault(tool_call.index, {"id": "", "name": "", "arguments": "" })
                if tool_call.id:
                    slot["id"] = tool_call.id
                if tool_call.function and tool_call.function.name:
                    slot["name"] = tool_call.function.name
                if tool_call.function and tool_call.function.arguments:
                    slot["arguments"] += tool_call.function.arguments
    tool_calls = None
    if tool_call_parts:
        tool_calls = [{
                "id": v["id"],
                "type": "function",
                "function": {"name": v["name"], "arguments": v["arguments"]}
            } for _, v in sorted(tool_call_parts.items())]
    msg = {
        "role": "assistant",
        "content": "".join(content_parts) or None,
        "tool_calls": tool_calls,
    }
    return msg, usage, finish_reason

async def stream_completion_async(client, *, model, messages, tools, tool_choice, max_tokens, on_text=None):
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=tool_choice,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True},
    )
    content_parts, tool_call_parts, usage, finish_reason = [], {}, None, None
    async for chunk in stream:
        if chunk.usage is not None:
            usage = chunk.usage
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason or finish_reason

        if delta.content:
            content_parts.append(delta.content)
            if on_text:
                ret = on_text(delta.content)
                if asyncio.iscoroutine(ret):
                    await ret
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                slot = tool_call_parts.setdefault(tool_call.index, {"id": "", "name": "", "arguments": "" })
                if tool_call.id:
                    slot["id"] = tool_call.id
                if tool_call.function and tool_call.function.name:
                    slot["name"] = tool_call.function.name
                if tool_call.function and tool_call.function.arguments:
                    slot["arguments"] += tool_call.function.arguments
    tool_calls = None
    if tool_call_parts:
        tool_calls = [{
                "id": v["id"],
                "type": "function",
                "function": {"name": v["name"], "arguments": v["arguments"]}
            } for _, v in sorted(tool_call_parts.items())]
    msg = {
        "role": "assistant",
        "content": "".join(content_parts) or None,
        "tool_calls": tool_calls,
    }
    return msg, usage, finish_reason

