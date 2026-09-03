"""
companion/server.py — 宋渡聊天前端后端
"""

import os
import json
import asyncio
import logging
import time
from pathlib import Path
from typing import Optional
from copy import deepcopy

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

STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(os.environ.get("COMPANION_DATA_DIR", str(Path(__file__).parent / "data")))
SETTINGS_FILE = DATA_DIR / "settings.json"
PROMPT_HISTORY_FILE = DATA_DIR / "prompt_history.json"

# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS = {
    "api_key": "",
    "model": "claude-sonnet-4-6",
    "system_prompt": "",
    "mcp_servers": [],
}


def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    _ensure_data_dir()
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text("utf-8"))
            merged = {**DEFAULT_SETTINGS, **data}
            return merged
        except Exception:
            logger.exception("Failed to load settings")
    return dict(DEFAULT_SETTINGS)


def save_settings(settings: dict):
    _ensure_data_dir()
    SETTINGS_FILE.write_text(json.dumps(settings, ensure_ascii=False, indent=2), "utf-8")


def load_prompt_history() -> list:
    if PROMPT_HISTORY_FILE.exists():
        try:
            return json.loads(PROMPT_HISTORY_FILE.read_text("utf-8"))
        except Exception:
            pass
    return []


def save_prompt_history(history: list):
    _ensure_data_dir()
    PROMPT_HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), "utf-8")


# ---------------------------------------------------------------------------
# MCP server tool discovery & execution
# ---------------------------------------------------------------------------

async def discover_tools(server: dict) -> list:
    """Fetch tool list from an MCP-compatible HTTP server."""
    url = server["url"].rstrip("/")
    async with httpx.AsyncClient(timeout=10) as http:
        try:
            resp = await http.get(f"{url}/api/tools")
            if resp.status_code == 200:
                tools = resp.json()
                if isinstance(tools, list):
                    return tools
            resp2 = await http.post(f"{url}/mcp", json={
                "jsonrpc": "2.0", "id": 1,
                "method": "tools/list", "params": {}
            })
            if resp2.status_code == 200:
                data = resp2.json()
                return data.get("result", {}).get("tools", [])
        except Exception as e:
            logger.warning(f"Tool discovery failed for {server.get('name','?')}: {e}")
    return []


async def call_mcp_tool(server: dict, tool_name: str, tool_input: dict) -> str:
    """Call a tool on an MCP server via HTTP."""
    url = server["url"].rstrip("/")
    async with httpx.AsyncClient(timeout=30) as http:
        try:
            resp = await http.post(f"{url}/api/tool/{tool_name}", json=tool_input)
            if resp.status_code == 200:
                return json.dumps(resp.json(), ensure_ascii=False)
            resp2 = await http.post(f"{url}/mcp", json={
                "jsonrpc": "2.0", "id": 1,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": tool_input}
            })
            if resp2.status_code == 200:
                data = resp2.json()
                result = data.get("result", data)
                return json.dumps(result, ensure_ascii=False)
            return json.dumps({"error": f"HTTP {resp.status_code}", "body": resp.text})
        except Exception as e:
            return json.dumps({"error": str(e)})


def build_tool_defs_and_router(settings: dict) -> tuple[list, dict]:
    """Return (tool_defs_for_claude, {tool_name: server_config})."""
    return [], {}


async def refresh_tools(settings: dict) -> tuple[list, dict]:
    """Discover tools from all configured MCP servers."""
    all_tools = []
    router = {}
    for server in settings.get("mcp_servers", []):
        if not server.get("enabled", True):
            continue
        tools = await discover_tools(server)
        prefix = server.get("prefix", "")
        for t in tools:
            name = t.get("name", "")
            if not name:
                continue
            tool_name = f"{prefix}{name}" if prefix else name
            tool_def = {
                "name": tool_name,
                "description": t.get("description", ""),
                "input_schema": t.get("inputSchema", t.get("input_schema",
                    {"type": "object", "properties": {}, "required": []})),
            }
            all_tools.append(tool_def)
            router[tool_name] = {"server": server, "original_name": name}
    return all_tools, router


# ---------------------------------------------------------------------------
# Settings API
# ---------------------------------------------------------------------------

async def get_settings(request: Request):
    settings = load_settings()
    safe = {**settings}
    if safe.get("api_key"):
        key = safe["api_key"]
        safe["api_key_preview"] = key[:8] + "…" + key[-4:] if len(key) > 12 else "***"
    else:
        safe["api_key_preview"] = ""
    safe.pop("api_key", None)
    return JSONResponse(safe)


async def update_settings(request: Request):
    body = await request.json()
    settings = load_settings()
    old_prompt = settings.get("system_prompt", "")

    if "api_key" in body:
        settings["api_key"] = body["api_key"]
    if "model" in body:
        settings["model"] = body["model"]
    if "system_prompt" in body:
        new_prompt = body["system_prompt"]
        if new_prompt != old_prompt:
            history = load_prompt_history()
            history.append({
                "prompt": old_prompt,
                "timestamp": time.time(),
                "label": body.get("prompt_label", ""),
            })
            if len(history) > 50:
                history = history[-50:]
            save_prompt_history(history)
        settings["system_prompt"] = new_prompt

    save_settings(settings)
    return JSONResponse({"ok": True})


async def get_prompt_history(request: Request):
    return JSONResponse(load_prompt_history())


# ---------------------------------------------------------------------------
# MCP servers management API
# ---------------------------------------------------------------------------

async def list_mcp_servers(request: Request):
    settings = load_settings()
    return JSONResponse(settings.get("mcp_servers", []))


async def add_mcp_server(request: Request):
    body = await request.json()
    name = body.get("name", "").strip()
    url = body.get("url", "").strip()
    if not name or not url:
        return JSONResponse({"error": "name and url required"}, status_code=400)

    settings = load_settings()
    server = {
        "name": name,
        "url": url,
        "prefix": body.get("prefix", ""),
        "enabled": True,
    }
    settings.setdefault("mcp_servers", []).append(server)
    save_settings(settings)
    return JSONResponse({"ok": True, "server": server})


async def remove_mcp_server(request: Request):
    body = await request.json()
    name = body.get("name", "")
    settings = load_settings()
    servers = settings.get("mcp_servers", [])
    settings["mcp_servers"] = [s for s in servers if s.get("name") != name]
    save_settings(settings)
    return JSONResponse({"ok": True})


async def toggle_mcp_server(request: Request):
    body = await request.json()
    name = body.get("name", "")
    settings = load_settings()
    for s in settings.get("mcp_servers", []):
        if s.get("name") == name:
            s["enabled"] = not s.get("enabled", True)
    save_settings(settings)
    return JSONResponse({"ok": True})


async def test_mcp_server(request: Request):
    body = await request.json()
    server = {"name": "test", "url": body.get("url", ""), "prefix": ""}
    tools = await discover_tools(server)
    return JSONResponse({"ok": True, "tools": tools})


# ---------------------------------------------------------------------------
# Chat endpoint — SSE streaming
# ---------------------------------------------------------------------------

async def chat_stream(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    settings = load_settings()
    system = body.get("system", settings.get("system_prompt", ""))
    api_key = settings.get("api_key", "")
    model = settings.get("model", "claude-sonnet-4-6")

    if not api_key:
        return JSONResponse({"error": "API key 未设置，请在设置中填写"}, status_code=400)

    tool_defs, router = await refresh_tools(settings)

    client = anthropic.AsyncAnthropic(api_key=api_key)

    async def event_generator():
        conversation = list(messages)
        max_tool_rounds = 10

        kwargs = dict(
            model=model,
            max_tokens=16000,
            messages=conversation,
            thinking={"type": "adaptive"},
        )
        if system:
            kwargs["system"] = system
        if tool_defs:
            kwargs["tools"] = tool_defs

        for _round in range(max_tool_rounds):
            try:
                async with client.messages.stream(**kwargs) as stream:
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

                for block in response.content:
                    if block.type == "tool_use":
                        tool_use_blocks.append(block)

                if response.stop_reason != "tool_use" or not tool_use_blocks:
                    yield f"data: {json.dumps({'type': 'done', 'stop_reason': response.stop_reason})}\n\n"
                    return

                conversation.append({"role": "assistant", "content": response.content})

                tool_results = []
                for tool_block in tool_use_blocks:
                    yield f"data: {json.dumps({'type': 'tool_exec', 'name': tool_block.name})}\n\n"
                    route = router.get(tool_block.name)
                    if route:
                        result = await call_mcp_tool(
                            route["server"], route["original_name"], tool_block.input
                        )
                    else:
                        result = json.dumps({"error": f"Unknown tool: {tool_block.name}"})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_block.id,
                        "content": result,
                    })
                    yield f"data: {json.dumps({'type': 'tool_result', 'name': tool_block.name})}\n\n"

                conversation.append({"role": "user", "content": tool_results})
                kwargs["messages"] = conversation

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
    settings = load_settings()
    return JSONResponse({
        "status": "ok",
        "model": settings.get("model", ""),
        "has_api_key": bool(settings.get("api_key")),
        "mcp_servers": len([s for s in settings.get("mcp_servers", []) if s.get("enabled", True)]),
    })


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = Starlette(
    routes=[
        Route("/", index),
        Route("/health", health),
        Route("/api/chat", chat_stream, methods=["POST"]),
        Route("/api/settings", get_settings, methods=["GET"]),
        Route("/api/settings", update_settings, methods=["PUT"]),
        Route("/api/prompt-history", get_prompt_history, methods=["GET"]),
        Route("/api/mcp-servers", list_mcp_servers, methods=["GET"]),
        Route("/api/mcp-servers", add_mcp_server, methods=["POST"]),
        Route("/api/mcp-servers/remove", remove_mcp_server, methods=["POST"]),
        Route("/api/mcp-servers/toggle", toggle_mcp_server, methods=["POST"]),
        Route("/api/mcp-servers/test", test_mcp_server, methods=["POST"]),
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
    port = int(os.environ.get("PORT", os.environ.get("COMPANION_PORT", "8766")))
    uvicorn.run(app, host="0.0.0.0", port=port)
