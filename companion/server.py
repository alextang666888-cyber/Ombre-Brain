"""
companion/server.py — 宋渡聊天前端后端
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Optional

import anthropic
import httpx
from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("companion")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OMBRE_BRAIN_URL = os.environ.get("OMBRE_BRAIN_URL", "http://localhost:8765")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL = os.environ.get("COMPANION_MODEL", "claude-sonnet-4-6")
STATIC_DIR = Path(__file__).parent / "static"

SYSTEM_PROMPT = os.environ.get("COMPANION_SYSTEM_PROMPT", "")

# ---------------------------------------------------------------------------
# Ombre-Brain MCP HTTP proxy — call breath / hold etc. via REST
# ---------------------------------------------------------------------------

TOOL_DEFS = [
    {
        "name": "breath",
        "description": "无参数,浮现权重最高的未消化记忆和核心准则。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "breath_search",
        "description": "按关键词/语义检索记忆。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "max_results": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "hold",
        "description": "存入一条新记忆。需要阿凌确认后才存。",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "记忆正文"},
                "importance": {"type": "number", "default": 5},
                "valence": {"type": "number", "default": 0},
                "arousal": {"type": "number", "default": 0},
                "domain": {"type": "string", "default": ""},
                "tags": {"type": "string", "default": ""},
            },
            "required": ["content"],
        },
    },
    {
        "name": "feel",
        "description": "查看当前情绪坐标。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "dream",
        "description": "触发记忆整合/梦境。",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "letter_read",
        "description": "读信件/日记。",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "日期 YYYY-MM-DD，留空读最新"},
            },
            "required": [],
        },
    },
    {
        "name": "letter_write",
        "description": "写日记。",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "日期 YYYY-MM-DD"},
                "content": {"type": "string", "description": "日记内容"},
            },
            "required": ["date", "content"],
        },
    },
    {
        "name": "I",
        "description": "写一条自我发现（候选态）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "自我发现内容"},
            },
            "required": ["content"],
        },
    },
]


async def call_ombre_tool(tool_name: str, tool_input: dict) -> str:
    """Call an Ombre-Brain tool via its HTTP API."""
    async with httpx.AsyncClient(timeout=30) as http:
        try:
            resp = await http.post(
                f"{OMBRE_BRAIN_URL}/api/tool/{tool_name}",
                json=tool_input,
            )
            if resp.status_code == 200:
                data = resp.json()
                return json.dumps(data, ensure_ascii=False)
            return json.dumps({"error": f"HTTP {resp.status_code}", "body": resp.text})
        except Exception as e:
            return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Chat endpoint — SSE streaming
# ---------------------------------------------------------------------------

async def chat_stream(request: Request):
    """SSE endpoint: receives messages, streams Claude response with tool use."""
    body = await request.json()
    messages = body.get("messages", [])
    system = body.get("system", SYSTEM_PROMPT)

    if not ANTHROPIC_API_KEY:
        return JSONResponse({"error": "ANTHROPIC_API_KEY not set"}, status_code=500)

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    async def event_generator():
        conversation = list(messages)
        max_tool_rounds = 10

        for _round in range(max_tool_rounds):
            try:
                async with client.messages.stream(
                    model=MODEL,
                    max_tokens=16000,
                    system=system,
                    messages=conversation,
                    tools=TOOL_DEFS,
                    thinking={"type": "adaptive"},
                ) as stream:
                    tool_use_blocks = []
                    current_text = ""

                    async for event in stream:
                        if event.type == "content_block_start":
                            if event.content_block.type == "text":
                                yield f"data: {json.dumps({'type': 'text_start'})}\n\n"
                            elif event.content_block.type == "thinking":
                                yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
                            elif event.content_block.type == "tool_use":
                                yield f"data: {json.dumps({'type': 'tool_start', 'name': event.content_block.name})}\n\n"

                        elif event.type == "content_block_delta":
                            if event.delta.type == "text_delta":
                                current_text += event.delta.text
                                yield f"data: {json.dumps({'type': 'text', 'text': event.delta.text})}\n\n"
                            elif event.delta.type == "thinking_delta":
                                yield f"data: {json.dumps({'type': 'thinking', 'text': event.delta.thinking})}\n\n"

                    response = stream.get_final_message()

                # Collect tool use blocks
                for block in response.content:
                    if block.type == "tool_use":
                        tool_use_blocks.append(block)

                if response.stop_reason != "tool_use" or not tool_use_blocks:
                    yield f"data: {json.dumps({'type': 'done', 'stop_reason': response.stop_reason})}\n\n"
                    return

                # Execute tools and continue conversation
                conversation.append({"role": "assistant", "content": response.content})

                tool_results = []
                for tool_block in tool_use_blocks:
                    yield f"data: {json.dumps({'type': 'tool_exec', 'name': tool_block.name})}\n\n"
                    result = await call_ombre_tool(tool_block.name, tool_block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": result,
                    })
                    yield f"data: {json.dumps({'type': 'tool_result', 'name': tool_block.name})}\n\n"

                conversation.append({"role": "user", "content": tool_results})

            except Exception as e:
                logger.exception("Stream error")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                return

        yield f"data: {json.dumps({'type': 'done', 'stop_reason': 'max_tool_rounds'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

async def index(request: Request):
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


async def health(request: Request):
    return JSONResponse({"status": "ok", "model": MODEL})


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Starlette(
    routes=[
        Route("/", index),
        Route("/health", health),
        Route("/api/chat", chat_stream, methods=["POST"]),
        Mount("/static", app=StaticFiles(directory=str(STATIC_DIR)), name="static"),
    ],
    middleware=[
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )
    ],
)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("COMPANION_PORT", "8766"))
    uvicorn.run(app, host="0.0.0.0", port=port)
