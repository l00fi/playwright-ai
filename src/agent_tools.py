"""
Browser tools for the LangGraph agent (see src/agent_core.py).

`build_browser_tools(env)` returns the list bound to the executor (navigate, click, type_text,
press_enter, dangerous_action). `finish_task` is included for API parity but is not used by the
subtask executor loop.

Module-level `env` and `tools` are the default singleton used by `agent_core`; tests can inject
a different env via `build_browser_tools` if needed.
"""

import time

from langchain_core.tools import tool

from src.entry import BrowserEnv


def build_browser_tools(env: BrowserEnv):
    """Return LangChain tools wired to the given BrowserEnv."""

    @tool
    def navigate(url: str) -> str:
        """Open a URL in the controlled page. The next agent step receives a fresh snapshot."""
        env.go_to(url)
        return f"Opened {url}. Vision will update; analyze the new page."

    @tool
    def click(element_id: int) -> str:
        """Click the overlay element with this numeric ID. Destructive targets are blocked here; use dangerous_action instead."""
        is_dangerous, details = env.is_dangerous_element(element_id)
        if is_dangerous:
            return (
                "Security layer: potentially destructive click blocked via click(). "
                f"{details} Use dangerous_action with an explicit description."
            )
        return env.click_element(element_id)

    @tool
    def type_text(element_id: int, text: str) -> str:
        """Type into the input or control identified by this overlay ID (replaces existing content as implemented by the env)."""
        return env.type_text(element_id, text)

    @tool
    def press_enter() -> str:
        """Press Enter in the page (e.g. submit a field or confirm inline)."""
        env.page.keyboard.press("Enter")
        try:
            env.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        time.sleep(1.0)
        return "Enter pressed; page may have updated."

    @tool
    def dangerous_action(element_id: int, action_description: str) -> str:
        """
        Use only for irreversible or high-impact actions (e.g. delete, pay, send, confirm destructive dialogs).
        Triggers a console prompt; the click runs only if the user types yes.
        """
        is_dangerous, details = env.is_dangerous_element(element_id)
        print("\n[SECURITY ALERT] Destructive action requested:")
        print(f"👉 Intent: {action_description}")
        if details:
            print(f"🔎 Element check: {details}")
        if not is_dangerous:
            print("⚠️ Element not flagged as obviously dangerous; dangerous_action was still requested.")
        user_input = input("Allow? Type 'yes' to confirm: ")
        if user_input.strip().lower() == "yes":
            result = env.click_element(element_id)
            return f"User approved. Result: {result}"
        return "User cancelled. Try another approach."

    @tool
    def finish_task(report: str) -> str:
        """End-of-run hook for a single-shot agent (not invoked by the subtask executor in agent_core)."""
        return f"Task finished. Report: {report}"

    return [navigate, click, type_text, press_enter, dangerous_action, finish_task]


env = BrowserEnv(headless=False)
tools = build_browser_tools(env)
