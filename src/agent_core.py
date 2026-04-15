from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, TypedDict
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src import prompts
from src.agent_tools import env, tools

load_dotenv()


def _env_int(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, "")
    try:
        return max(int(raw), min_value)
    except Exception:
        return default


def _llm_invoke(llm: Any, messages: list):
    """
    Wrap LangChain invoke: OpenRouter/network failures sometimes yield a None raw response;
    langchain-openai then crashes with AttributeError: 'NoneType' object has no attribute 'model_dump'.
    """
    try:
        return llm.invoke(messages)
    except AttributeError as e:
        err = str(e)
        if "NoneType" in err and "model_dump" in err:
            raise RuntimeError(
                "LLM API returned an empty or invalid response (nothing to parse into a chat result). "
                "Typical causes: missing/invalid OPENROUTER_API_KEY, network issues, OpenRouter outage, "
                "or model/billing limits for AGENT_MODEL. Verify .env and https://openrouter.ai/ status."
            ) from e
        raise


def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_json_candidate(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if "```" in raw:
        for chunk in raw.split("```"):
            candidate = chunk.strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
            if candidate.startswith("{") or candidate.startswith("["):
                return candidate
    first_obj = raw.find("{")
    last_obj = raw.rfind("}")
    if first_obj != -1 and last_obj != -1 and first_obj < last_obj:
        return raw[first_obj : last_obj + 1]
    first_arr = raw.find("[")
    last_arr = raw.rfind("]")
    if first_arr != -1 and last_arr != -1 and first_arr < last_arr:
        return raw[first_arr : last_arr + 1]
    return raw


def _tool_signature(name: str, args: dict) -> str:
    try:
        payload = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except Exception:
        payload = str(args)
    return f"{name}:{payload}"


def _looks_success(result_text: str) -> bool:
    low = result_text.lower()
    return not any(x in low for x in ["ошибка", "error", "failed", "not found", "отмен", "blocked"])


def _host_key(netloc: str) -> str:
    h = (netloc or "").lower().strip()
    if h.startswith("www."):
        h = h[4:]
    return h


def _maybe_navigation_done(subtask: str, last_action: ActionRecord, page_text: str) -> dict[str, Any] | None:
    """If subtask is clearly 'open/go to URL' and navigate reached that host, mark done without LLM."""
    if last_action.tool != "navigate" or not last_action.success:
        return None
    url_arg = str(last_action.args.get("url", "")).strip()
    if "http" not in url_arg.lower():
        return None
    try:
        nav_host = _host_key(urlparse(url_arg).netloc)
    except Exception:
        return None
    if not nav_host:
        return None

    st = subtask.strip()
    st_low = st.lower()
    nav_keywords = (
        "open ",
        "go to ",
        "navigate",
        "visit",
        "перейти",
        "открыть",
        "зайти",
        "site",
        "сайт",
        "url",
    )
    urls_in_subtask = re.findall(r"https?://[^\s\]\)\"']+", st)
    has_nav_intent = bool(urls_in_subtask) or any(k in st_low for k in nav_keywords)
    if not has_nav_intent:
        return None

    expected_hosts: list[str] = []
    for u in urls_in_subtask:
        try:
            h = _host_key(urlparse(u).netloc)
            if h:
                expected_hosts.append(h)
        except Exception:
            pass
    for m in re.finditer(r"\b([a-z0-9-]+(?:\.[a-z0-9-]+)+)\b", st, re.I):
        dom = m.group(1).lower()
        if dom not in ("http", "https") and "." in dom:
            expected_hosts.append(dom)

    try:
        cur_host = _host_key(urlparse(last_action.url).netloc)
    except Exception:
        cur_host = ""

    for eh in expected_hosts:
        if eh and (eh == cur_host or (cur_host and (cur_host.endswith(eh) or eh in cur_host))):
            return {
                "status": "done",
                "reason": f"Navigation subtask satisfied: reached host {cur_host or nav_host}.",
            }
    if urls_in_subtask and nav_host and cur_host and (nav_host == cur_host or nav_host in cur_host):
        return {"status": "done", "reason": f"Navigation completed to {cur_host or nav_host}."}

    # Fallback: hostname appears in visible text (title/URL line in parse output)
    if nav_host and nav_host in (page_text or "").lower():
        return {"status": "done", "reason": f"Navigation landed on {nav_host} (host visible in page context)."}
    return None


def _evidence_grounds_in_page(evidence: str, page_text: str, last_action: ActionRecord) -> bool:
    ev = (evidence or "").strip()
    if len(ev) < 8:
        return False
    pl = (page_text or "").lower()
    el = ev.lower()
    if el in pl:
        return True
    for n in range(0, max(1, len(el) - 24)):
        chunk = el[n : n + 32]
        if len(chunk) >= 12 and chunk in pl:
            return True
    if last_action.tool == "navigate":
        try:
            u = str(last_action.args.get("url", ""))
            host = _host_key(urlparse(u).netloc)
            combined = f"{page_text} {last_action.url}".lower()
            if host and host in combined:
                return True
        except Exception:
            pass
    return False


def _policy_validate_done(
    page_text: str,
    last_action: ActionRecord,
    status: str,
    reason: str,
    evidence: str,
) -> tuple[str, str]:
    if status != "done":
        return status, reason
    pt = (page_text or "").strip()
    if len(pt) < 25 and "SNAPSHOT_DEGRADED" not in pt:
        return "progress", "Policy: cannot confirm done — page text too short to verify."
    if not _evidence_grounds_in_page(evidence, page_text, last_action):
        return (
            "progress",
            "Policy: critic marked done but evidence is not grounded in visible page text.",
        )
    return status, reason


class RunLogger:
    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_dir = os.path.join("artifacts", "runs", self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.run_dir, "events.jsonl")
        self.md_path = os.path.join(self.run_dir, "timeline.md")
        with open(self.md_path, "w", encoding="utf-8") as f:
            f.write(f"# Run timeline {self.run_id}\n\n")

    def log(self, event_type: str, **payload):
        event = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event_type, **payload}
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        summary = payload.get("summary", "") or ", ".join(
            f"{k}={v}" for k, v in payload.items() if k not in {"raw", "result", "text_state"}
        )
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(f"- `{event['ts']}` **{event_type}**: {summary}\n")


@dataclass
class ActionRecord:
    subtask_idx: int
    step: int
    tool: str
    args: dict[str, Any]
    result: str
    signature: str
    success: bool
    url: str


class Critic:
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

    def _invoke_json(self, system_text: str, user_text: str, max_attempts: int = 3):
        full_system_text = f"{prompts.CRITIC_JSON_SYSTEM_PROMPT}\n\n{system_text}"
        base = [SystemMessage(content=full_system_text), HumanMessage(content=user_text)]
        response = _llm_invoke(self.llm, base)
        for _ in range(max_attempts):
            raw = str(response.content or "")
            parsed = _safe_json_loads(_extract_json_candidate(raw))
            if parsed is not None:
                return parsed, raw
            response = _llm_invoke(self.llm, base + [HumanMessage(content=prompts.CRITIC_JSON_REPAIR_PROMPT)])
        return None, str(response.content or "")

    def make_plan(self, goal: str) -> list[str]:
        parsed, _ = self._invoke_json(
            system_text=prompts.PLAN_SYSTEM_PROMPT,
            user_text=prompts.PLAN_USER_PROMPT_TEMPLATE.format(goal=goal),
        )
        if isinstance(parsed, list):
            cleaned = [str(x).strip() for x in parsed if str(x).strip()]
            if cleaned:
                return cleaned[:8]
        return list(prompts.GENERIC_FALLBACK_PLAN)

    def detect_loop(self, local_signatures: list[str], repeat_threshold: int) -> str | None:
        if len(local_signatures) < repeat_threshold:
            return None
        recent = local_signatures[-repeat_threshold:]
        if len(set(recent)) == 1:
            return recent[-1]
        return None

    def evaluate_subtask_progress(
        self,
        goal: str,
        subtask: str,
        last_action: ActionRecord,
        page_text: str,
        recent_actions: list[ActionRecord],
    ) -> dict[str, Any]:
        if not last_action.success:
            return {"status": "stuck", "reason": "Last action failed."}

        nav_done = _maybe_navigation_done(subtask, last_action, page_text)
        if nav_done is not None:
            return nav_done

        recent_summary = "\n".join(
            f"- {a.tool} {a.args} => {a.result[:120]}" for a in recent_actions[-4:]
        ) or "- no actions"
        prompt = prompts.CRITIC_EVAL_USER_PROMPT_TEMPLATE.format(
            goal=goal,
            subtask=subtask,
            recent_summary=recent_summary,
            page_text=page_text[:2500],
        )
        parsed, raw = self._invoke_json(system_text=prompts.CRITIC_EVAL_SYSTEM_PROMPT, user_text=prompt)
        if isinstance(parsed, dict) and parsed.get("status") in {"done", "progress", "stuck", "offtrack"}:
            status = str(parsed.get("status"))
            reason = str(parsed.get("reason", ""))
            scope_ok = bool(parsed.get("scope_ok", True))
            evidence = str(parsed.get("evidence", "")).strip()

            # Strict scope guard: do not allow "done" if critic itself reports scope mismatch.
            if status == "done" and not scope_ok:
                status = "progress"
                reason = f"Scope mismatch for current subtask. {reason}".strip()

            if status == "done":
                status, reason = _policy_validate_done(page_text, last_action, status, reason, evidence)

            if evidence:
                reason = f"{reason} | evidence: {evidence}".strip()
            return {"status": status, "reason": reason}
        return {"status": "stuck", "reason": f"Invalid critic JSON response: {raw[:180]}"}

    def extract_answer(self, goal: str, last_page_text: str) -> str:
        if not (last_page_text or "").strip():
            return ""
        user_text = prompts.EXTRACT_ANSWER_USER_TEMPLATE.format(
            goal=goal,
            page_text=last_page_text[:6000],
        )
        response = _llm_invoke(
            self.llm,
            [
                SystemMessage(content=prompts.EXTRACT_ANSWER_SYSTEM_PROMPT),
                HumanMessage(content=user_text),
            ],
        )
        return str(response.content or "").strip()

    def final_report(
        self,
        goal: str,
        plan: list[str],
        completed_count: int,
        action_log: list[ActionRecord],
        blocked_reason: str = "",
        extracted_answer: str = "",
    ) -> str:
        status = "SUCCESS" if completed_count == len(plan) and not blocked_reason else "BLOCKED"
        actions_tail = "\n".join(
            f"- [{a.subtask_idx + 1}.{a.step}] {a.tool} {a.args} -> {a.result[:120]}" for a in action_log[-12:]
        ) or "- no actions"
        done_list = "\n".join(f"- {'[x]' if i < completed_count else '[ ]'} {task}" for i, task in enumerate(plan))
        report = (
            f"status={status}\n"
            f"goal={goal}\n"
            f"completed_subtasks={completed_count}/{len(plan)}\n"
            f"plan:\n{done_list}\n"
            f"recent_actions:\n{actions_tail}"
        )
        if extracted_answer:
            report += f"\nextracted_answer={extracted_answer}"
        if blocked_reason:
            report += f"\nblocker={blocked_reason}"
            if (extracted_answer or "").strip():
                report += (
                    "\nintegrity_note=Run stopped before all planned subtasks completed. "
                    "extracted_answer is inferred only from the last snapshot text, not from a "
                    "fully verified navigation sequence; treat it as best-effort."
                )
        return report


class Executor:
    def __init__(self, llm: ChatOpenAI):
        self.llm_with_tools = llm.bind_tools([t for t in tools if t.name != "finish_task"])

    def decide_next_action(
        self,
        goal: str,
        subtask: str,
        text_metadata: str,
        screenshot_path: str,
        recent_actions: list[ActionRecord],
        critic_hint: str = "",
    ):
        with open(screenshot_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        recent = "\n".join(f"- {a.tool} {a.args} => {a.result[:100]}" for a in recent_actions[-5:]) or "- no actions"
        instruction = prompts.EXECUTOR_USER_PROMPT_TEMPLATE.format(
            goal=goal,
            subtask=subtask,
            critic_hint=(critic_hint or "none"),
            recent_actions=recent,
            text_state=text_metadata[:4000],
        )
        messages = [
            SystemMessage(content=prompts.EXECUTOR_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                ]
            ),
        ]
        response = _llm_invoke(self.llm_with_tools, messages)
        if not response.tool_calls:
            response = _llm_invoke(
                self.llm_with_tools,
                messages + [HumanMessage(content=prompts.EXECUTOR_FORMAT_REPAIR_PROMPT)],
            )
        return response

    def execute_tool_call(self, tool_call: dict):
        tool_name = tool_call["name"]
        args = tool_call.get("args", {})
        action = next((t for t in tools if t.name == tool_name), None)
        if action is None:
            return f"Error: unknown tool '{tool_name}'."
        return str(action.invoke(args))


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("AGENT_MODEL", "openai/gpt-4o"),
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )


class RuntimeState(TypedDict):
    goal: str
    plan: list[str]
    current_subtask_idx: int
    completed_count: int
    local_steps: int
    restarts: int
    critic_hint: str
    total_steps: int
    blocked_reason: str
    status: str
    action_log: list[dict]
    local_action_signatures: list[str]
    recent_subtask_actions: list[dict]
    text_metadata: str
    screenshot_path: str
    post_text_metadata: str
    pending_tool_call: dict
    last_action: dict
    final_report: str
    snapshot_fail_streak: int
    last_page_text: str


def build_runtime_app(critic: Critic, executor: Executor, run_logger: RunLogger):
    max_total_steps = _env_int("AGENT_MAX_TOTAL_STEPS", 60, min_value=10)
    max_steps_per_subtask = _env_int("AGENT_MAX_STEPS_PER_SUBTASK", 10, min_value=2)
    max_restarts_per_subtask = _env_int("AGENT_MAX_RESTARTS_PER_SUBTASK", 2, min_value=0)
    loop_repeat_threshold = _env_int("AGENT_LOOP_REPEAT_THRESHOLD", 3, min_value=2)
    max_snapshot_fail_streak = _env_int("AGENT_MAX_SNAPSHOT_FAIL_STREAK", 5, min_value=1)

    def _action_from_dict(data: dict) -> ActionRecord:
        return ActionRecord(**data)

    def init_node(state: RuntimeState):
        goal = state.get("goal", "").strip()
        plan = critic.make_plan(goal) if goal else []
        run_logger.log("goal_received", goal=goal, summary=f"goal='{goal[:180]}'")
        run_logger.log("plan_created", subtasks=plan, summary=f"Critic created {len(plan)} subtasks.")
        if plan:
            print("\n📋 [Critic] execution plan:")
            for i, task in enumerate(plan, start=1):
                print(f"  {i}. {task}")
        return {
            "plan": plan,
            "current_subtask_idx": 0,
            "completed_count": 0,
            "local_steps": 0,
            "restarts": 0,
            "critic_hint": "",
            "total_steps": 0,
            "blocked_reason": "",
            "status": "RUNNING",
            "action_log": [],
            "local_action_signatures": [],
            "recent_subtask_actions": [],
            "final_report": "",
            "snapshot_fail_streak": 0,
            "last_page_text": "",
        }

    def subtask_setup_node(state: RuntimeState):
        idx = state["current_subtask_idx"]
        if idx >= len(state["plan"]):
            return {"status": "SUCCESS"}
        subtask = state["plan"][idx]
        run_logger.log("subtask_started", subtask_idx=idx, subtask=subtask, summary=f"Start subtask {idx + 1}: {subtask}")
        return {"local_steps": 0, "restarts": 0, "critic_hint": "", "local_action_signatures": [], "recent_subtask_actions": []}

    def guard_node(state: RuntimeState):
        if state.get("status") in {"SUCCESS", "BLOCKED"}:
            return {}
        # Consume restart marker once and continue normal execution flow.
        if state.get("status") == "RESTART_SUBTASK":
            return {"status": "RUNNING"}
        if state["total_steps"] >= max_total_steps:
            reason = f"Reached total_steps limit={max_total_steps}"
            run_logger.log("critic_action_block", reason=reason, summary=reason)
            return {"status": "BLOCKED", "blocked_reason": reason}
        if state["local_steps"] >= max_steps_per_subtask:
            restarts = state["restarts"] + 1
            if restarts > max_restarts_per_subtask:
                reason = f"Subtask {state['current_subtask_idx'] + 1} exceeded restart limit ({max_restarts_per_subtask})."
                run_logger.log("critic_action_block", reason=reason, summary=reason)
                return {"status": "BLOCKED", "blocked_reason": reason}
            run_logger.log(
                "critic_action_restart",
                subtask_idx=state["current_subtask_idx"],
                restart_no=restarts,
                reason="max_steps_per_subtask exceeded",
                summary=f"Restart subtask {state['current_subtask_idx'] + 1} due to step limit.",
            )
            return {
                "status": "RESTART_SUBTASK",
                "restarts": restarts,
                "local_steps": 0,
                "critic_hint": "Switch strategy; do not repeat prior actions.",
                "local_action_signatures": [],
                "recent_subtask_actions": [],
            }
        return {}

    def snapshot_node(state: RuntimeState):
        shot, text = env.get_visual_state()
        streak = int(state.get("snapshot_fail_streak", 0)) + 1
        if not shot:
            run_logger.log(
                "snapshot_failed",
                subtask_idx=state["current_subtask_idx"],
                step=state["local_steps"] + 1,
                reason=text,
                streak=streak,
                summary=f"Snapshot failed ({streak}/{max_snapshot_fail_streak}): {text}",
            )
            if streak >= max_snapshot_fail_streak:
                reason = f"Snapshot failed {streak} times in a row: {text}"
                run_logger.log("critic_action_block", reason=reason, summary=reason)
                return {
                    "status": "BLOCKED",
                    "blocked_reason": reason,
                    "text_metadata": text,
                    "snapshot_fail_streak": streak,
                }
            return {"status": "SNAPSHOT_FAILED", "text_metadata": text, "snapshot_fail_streak": streak}
        return {"screenshot_path": shot, "text_metadata": text, "snapshot_fail_streak": 0}

    def executor_decide_node(state: RuntimeState):
        idx = state["current_subtask_idx"]
        subtask = state["plan"][idx]
        recent = [_action_from_dict(a) for a in state.get("recent_subtask_actions", [])]
        response = executor.decide_next_action(
            goal=state["goal"],
            subtask=subtask,
            text_metadata=state["text_metadata"],
            screenshot_path=state["screenshot_path"],
            recent_actions=recent,
            critic_hint=state.get("critic_hint", ""),
        )
        reasoning = str(getattr(response, "content", "") or "").strip()
        if reasoning:
            print(f"\n💭 [Executor reasoning] {reasoning[:900]}{'…' if len(reasoning) > 900 else ''}")
        tool_calls = list(getattr(response, "tool_calls", []) or [])
        if not tool_calls:
            run_logger.log(
                "executor_no_tool_call",
                subtask_idx=idx,
                step=state["local_steps"] + 1,
                summary="Executor returned no tool call.",
            )
            return {"status": "DECIDE_FAILED", "critic_hint": "Executor produced no tool call."}

        tool_call = tool_calls[0]
        run_logger.log(
            "executor_tool_selected",
            subtask_idx=idx,
            step=state["local_steps"] + 1,
            tool=tool_call["name"],
            args=tool_call.get("args", {}),
            summary=f"Executor selected {tool_call['name']} at step {state['local_steps'] + 1}.",
        )
        print(f"\n🤖 [Executor] subtask={idx + 1} step={state['local_steps'] + 1} -> {tool_call['name']} {tool_call.get('args', {})}")
        return {"pending_tool_call": tool_call}

    def executor_execute_node(state: RuntimeState):
        tool_call = state.get("pending_tool_call", {})
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        signature = _tool_signature(tool_name, args)
        result = executor.execute_tool_call(tool_call) if tool_name else "Error: no pending tool call."

        try:
            current_url = env.page.url
        except Exception:
            current_url = ""

        local_step = state["local_steps"] + 1
        action = ActionRecord(
            subtask_idx=state["current_subtask_idx"],
            step=local_step,
            tool=tool_name or "none",
            args=args,
            result=result,
            signature=signature,
            success=_looks_success(result),
            url=current_url,
        )

        run_logger.log(
            "executor_tool_result",
            subtask_idx=action.subtask_idx,
            step=action.step,
            tool=action.tool,
            args=action.args,
            result=action.result,
            success=action.success,
            url=action.url,
            summary=f"Tool result {action.tool}: {'ok' if action.success else 'fail'}",
        )
        print(f"   ↳ result: {'ok' if action.success else 'fail'} | url: {action.url}")

        post_shot, post_text = env.get_visual_state()
        if not post_shot:
            post_text = f"[WARNING] post-action snapshot unavailable: {post_text}\n\nfallback_pre_action_state:\n{state['text_metadata']}"
            run_logger.log(
                "post_snapshot_failed",
                subtask_idx=state["current_subtask_idx"],
                step=action.step,
                summary="Post-action snapshot unavailable; fallback used.",
            )

        action_log = list(state["action_log"]) + [asdict(action)]
        recent_actions = list(state.get("recent_subtask_actions", [])) + [asdict(action)]
        signatures = list(state.get("local_action_signatures", [])) + [signature]

        return {
            "last_action": asdict(action),
            "action_log": action_log,
            "recent_subtask_actions": recent_actions,
            "local_action_signatures": signatures,
            "local_steps": local_step,
            "total_steps": state["total_steps"] + 1,
            "post_text_metadata": post_text,
            "last_page_text": post_text[:8000] if isinstance(post_text, str) else "",
        }

    def critic_evaluate_node(state: RuntimeState):
        if state.get("status") in {"SUCCESS", "BLOCKED"}:
            return {}

        loop_sig = critic.detect_loop(state.get("local_action_signatures", []), repeat_threshold=loop_repeat_threshold)
        if loop_sig:
            restarts = state["restarts"] + 1
            if restarts > max_restarts_per_subtask:
                reason = (
                    f"Subtask {state['current_subtask_idx'] + 1} looped ({loop_sig}) "
                    f"and exceeded restart limit."
                )
                run_logger.log("critic_action_block", reason=reason, summary=reason)
                return {"status": "BLOCKED", "blocked_reason": reason}
            run_logger.log(
                "critic_action_restart",
                subtask_idx=state["current_subtask_idx"],
                restart_no=restarts,
                reason=f"loop_detected: {loop_sig}",
                summary=f"Restart subtask {state['current_subtask_idx'] + 1}: loop detected.",
            )
            return {
                "status": "RESTART_SUBTASK",
                "restarts": restarts,
                "local_steps": 0,
                "critic_hint": "Loop detected. Choose a radically different strategy.",
                "local_action_signatures": [],
                "recent_subtask_actions": [],
            }

        last_action = _action_from_dict(state["last_action"])
        recent_actions = [_action_from_dict(a) for a in state.get("recent_subtask_actions", [])]
        subtask = state["plan"][state["current_subtask_idx"]]
        check = critic.evaluate_subtask_progress(
            goal=state["goal"],
            subtask=subtask,
            last_action=last_action,
            page_text=state.get("post_text_metadata", ""),
            recent_actions=recent_actions,
        )
        status = check.get("status", "progress")
        reason = str(check.get("reason", ""))
        run_logger.log(
            "critic_reaction",
            subtask_idx=state["current_subtask_idx"],
            step=last_action.step,
            status=status,
            reason=reason,
            summary=f"Critic: status={status}, reason={reason[:140]}",
        )
        print(f"🧩 [Critic] status={status} reason={reason}")

        if status == "done":
            next_idx = state["current_subtask_idx"] + 1
            completed = state["completed_count"] + 1
            run_logger.log(
                "subtask_completed",
                subtask_idx=state["current_subtask_idx"],
                completed_count=completed,
                summary=f"Subtask {state['current_subtask_idx'] + 1} completed.",
            )
            print(f"✅ [Critic] subtask {state['current_subtask_idx'] + 1} completed")
            if next_idx >= len(state["plan"]):
                return {"status": "SUCCESS", "completed_count": completed}
            return {
                "status": "NEXT_SUBTASK",
                "completed_count": completed,
                "current_subtask_idx": next_idx,
                "local_steps": 0,
                "restarts": 0,
                "critic_hint": "",
                "local_action_signatures": [],
                "recent_subtask_actions": [],
            }

        if status in {"stuck", "offtrack"}:
            restarts = state["restarts"] + 1
            if restarts > max_restarts_per_subtask:
                reason_block = (
                    f"Critic stopped on subtask {state['current_subtask_idx'] + 1}: {reason}. "
                    "Restart limit exceeded."
                )
                run_logger.log("critic_action_block", reason=reason_block, summary=reason_block)
                return {"status": "BLOCKED", "blocked_reason": reason_block}
            run_logger.log(
                "critic_action_restart",
                subtask_idx=state["current_subtask_idx"],
                restart_no=restarts,
                reason=reason,
                summary=f"Restart subtask {state['current_subtask_idx'] + 1}: {reason[:140]}",
            )
            print(f"🔁 [Critic] restarting subtask {state['current_subtask_idx'] + 1}: {reason}")
            return {
                "status": "RESTART_SUBTASK",
                "restarts": restarts,
                "local_steps": 0,
                "critic_hint": f"Restart reason: {reason}. Switch approach.",
                "local_action_signatures": [],
                "recent_subtask_actions": [],
            }

        return {"status": "RUNNING", "critic_hint": reason}

    def finalize_node(state: RuntimeState):
        action_log = [_action_from_dict(a) for a in state.get("action_log", [])]
        extracted = ""
        try:
            extracted = critic.extract_answer(state["goal"], state.get("last_page_text", ""))
        except Exception as e:
            run_logger.log("extract_answer_failed", error=str(e)[:200], summary="extract_answer failed")
        report = critic.final_report(
            goal=state["goal"],
            plan=state["plan"],
            completed_count=state["completed_count"],
            action_log=action_log,
            blocked_reason=state.get("blocked_reason", ""),
            extracted_answer=extracted,
        )
        blocked = state.get("status") == "BLOCKED"
        run_logger.log(
            "run_finished",
            completed_subtasks=state["completed_count"],
            total_subtasks=len(state["plan"]),
            blocked_reason=state.get("blocked_reason", ""),
            report=report,
            answer_extracted=bool((extracted or "").strip()),
            partial_text_answer=bool(blocked and (extracted or "").strip()),
            summary=(
                f"Run finished: {state['completed_count']}/{len(state['plan'])}, blocked={blocked}; "
                f"best_effort_answer={'yes' if blocked and (extracted or '').strip() else 'no'}"
            ),
        )
        return {"final_report": report}

    def route_after_guard(state: RuntimeState):
        status = state.get("status", "RUNNING")
        if status in {"SUCCESS", "BLOCKED"}:
            return "finalize"
        if status == "RESTART_SUBTASK":
            return "guard"
        return "snapshot"

    def route_after_snapshot(state: RuntimeState):
        if state.get("status") == "BLOCKED":
            return "finalize"
        if state.get("status") == "SNAPSHOT_FAILED":
            return "guard"
        return "executor_decide"

    def route_after_decide(state: RuntimeState):
        if state.get("status") == "DECIDE_FAILED":
            return "guard"
        return "executor_execute"

    def route_after_critic(state: RuntimeState):
        status = state.get("status", "RUNNING")
        if status in {"SUCCESS", "BLOCKED"}:
            return "finalize"
        if status == "NEXT_SUBTASK":
            return "subtask_setup"
        return "guard"

    graph = StateGraph(RuntimeState)
    graph.add_node("init", init_node)
    graph.add_node("subtask_setup", subtask_setup_node)
    graph.add_node("guard", guard_node)
    graph.add_node("snapshot", snapshot_node)
    graph.add_node("executor_decide", executor_decide_node)
    graph.add_node("executor_execute", executor_execute_node)
    graph.add_node("critic_evaluate", critic_evaluate_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "subtask_setup")
    graph.add_edge("subtask_setup", "guard")
    graph.add_conditional_edges("guard", route_after_guard, {"snapshot": "snapshot", "guard": "guard", "finalize": "finalize"})
    graph.add_conditional_edges(
        "snapshot",
        route_after_snapshot,
        {"executor_decide": "executor_decide", "guard": "guard", "finalize": "finalize"},
    )
    graph.add_conditional_edges("executor_decide", route_after_decide, {"executor_execute": "executor_execute", "guard": "guard"})
    graph.add_edge("executor_execute", "critic_evaluate")
    graph.add_conditional_edges(
        "critic_evaluate",
        route_after_critic,
        {"guard": "guard", "subtask_setup": "subtask_setup", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def run_app():
    critic = Critic(_build_llm())
    executor = Executor(_build_llm())
    run_logger = RunLogger()
    app = build_runtime_app(critic, executor, run_logger)

    print("=" * 60)
    print("🚀 Web-Agent (Critic + Executor + LangGraph) started")
    print("=" * 60)
    print("🌐 Opening start page...")
    env.go_to("https://ya.ru")

    goal = input("\n📝 Enter user task: ").strip()
    if not goal:
        print("Empty task. Stopping.")
        run_logger.log("run_stopped", reason="empty_goal", summary="Empty goal; run stopped.")
        env.close()
        return

    initial_state: RuntimeState = {
        "goal": goal,
        "plan": [],
        "current_subtask_idx": 0,
        "completed_count": 0,
        "local_steps": 0,
        "restarts": 0,
        "critic_hint": "",
        "total_steps": 0,
        "blocked_reason": "",
        "status": "RUNNING",
        "action_log": [],
        "local_action_signatures": [],
        "recent_subtask_actions": [],
        "text_metadata": "",
        "screenshot_path": "",
        "post_text_metadata": "",
        "pending_tool_call": {},
        "last_action": {},
        "final_report": "",
        "snapshot_fail_streak": 0,
        "last_page_text": "",
    }

    recursion_limit = _env_int("AGENT_LANGGRAPH_RECURSION_LIMIT", 200, min_value=30)
    result = app.invoke(initial_state, config={"recursion_limit": recursion_limit})
    print("\n" + "=" * 60)
    print("📄 Final critic report:")
    print(result.get("final_report", "No report."))
    print("=" * 60)
    print(f"🗂️ Runtime logs: {run_logger.run_dir}")
    env.close()
