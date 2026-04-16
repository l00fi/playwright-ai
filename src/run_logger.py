from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any


class RunLogger:
    """Structured run logger with timeline, route trace, and runtime metrics."""

    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join("artifacts", "runs", self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.run_dir, "events.jsonl")
        self.md_path = os.path.join(self.run_dir, "timeline.md")
        self.started_at = time.perf_counter()
        self.tool_call_counts: dict[str, int] = {}
        self.agent_route: list[dict[str, Any]] = []
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write(f"# Run timeline {self.run_id}\n\n")

    def log(self, event_type: str, **payload):
        """
        Append one structured event and update derived counters.

        Side effects:
        - writes JSONL (`events.jsonl`) for machine parsing,
        - writes Markdown timeline (`timeline.md`) for quick human review,
        - updates in-memory route and per-tool counters.
        """
        if event_type == "executor_tool_result":
            tool = str(payload.get("tool", "")).strip() or "unknown"
            self.tool_call_counts[tool] = int(self.tool_call_counts.get(tool, 0)) + 1
            route_step = {
                "subtask_idx": payload.get("subtask_idx"),
                "step": payload.get("step"),
                "tool": tool,
                "url": payload.get("url", ""),
            }
            # Keep a compact ordered route with de-duplication of identical consecutive states.
            if not self.agent_route or self.agent_route[-1] != route_step:
                self.agent_route.append(route_step)

        event = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event_type, **payload}
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        summary = payload.get("summary", "") or ", ".join(
            f"{k}={v}" for k, v in payload.items() if k not in {"raw", "result", "text_state"}
        )
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(f"- `{event['ts']}` **{event_type}**: {summary}\n")

    def runtime_stats(self) -> dict[str, Any]:
        """Return run-level derived metrics for final report logging."""
        elapsed_s = max(0.0, time.perf_counter() - self.started_at)
        return {
            "elapsed_seconds": round(elapsed_s, 3),
            "tool_call_counts": dict(sorted(self.tool_call_counts.items())),
            "agent_route": list(self.agent_route),
        }

