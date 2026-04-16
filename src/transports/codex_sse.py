from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass
class CodexSSEEvent:
    payload: dict
    raw_line: str


class CodexHttpError(RuntimeError):
    pass


def stream_sse_events(
    *,
    url: str,
    headers: dict[str, str],
    payload: dict,
    timeout_seconds: int = 60,
) -> Iterator[CodexSSEEvent]:
    request = Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            for raw_bytes in response:
                raw_line = raw_bytes.decode("utf-8", errors="replace").strip()
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data_text = raw_line[len("data:"):].strip()
                if data_text == "[DONE]":
                    return
                try:
                    payload_obj = json.loads(data_text)
                except json.JSONDecodeError:
                    continue
                yield CodexSSEEvent(payload=payload_obj, raw_line=raw_line)
    except HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise CodexHttpError(
            f"Codex HTTP error {exc.code}: {error_text}"
        ) from exc
    except URLError as exc:
        raise CodexHttpError(f"Codex connection error: {exc}") from exc
