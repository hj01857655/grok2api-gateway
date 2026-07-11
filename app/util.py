from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def now_ts() -> int:
    return int(time.time())


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def parse_json(s: str) -> Optional[Any]:
    try:
        return json.loads(s)
    except Exception:
        return None


def sse_data(obj: Any) -> str:
    """OpenAI-style SSE: data-only frame."""
    if isinstance(obj, str):
        return f"data: {obj}\n\n"
    return f"data: {dumps(obj)}\n\n"


def sse_event(event: str, obj: Any) -> str:
    """Anthropic / Responses style: event + data frame."""
    payload = obj if isinstance(obj, str) else dumps(obj)
    return f"event: {event}\ndata: {payload}\n\n"


async def iter_sse_data_lines(raw_stream: AsyncIterator[bytes]) -> AsyncIterator[str]:
    """Parse SSE stream into data payload strings (no 'data:' prefix)."""
    buf = ""
    async for chunk in raw_stream:
        if chunk.startswith(b"__HTTP_ERROR__"):
            yield "__HTTP_ERROR__" + chunk.decode("utf-8", errors="replace")[len("__HTTP_ERROR__") :]
            return
        buf += chunk.decode("utf-8", errors="replace")
        while "\n" in buf:
            line, buf = buf.split("\n", 1)
            line = line.rstrip("\r")
            if not line:
                continue
            if line.startswith("data:"):
                yield line[5:].lstrip()
