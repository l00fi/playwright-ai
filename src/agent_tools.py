import time

from langchain_core.tools import tool

from src.entry import BrowserEnv


def build_browser_tools(env: BrowserEnv):
    """Factory: all tools bound to a given BrowserEnv (tests or DI)."""

    @tool
    def navigate(url: str) -> str:
        """Navigate to the given URL; vision refreshes on the next step."""
        env.go_to(url)
        return f"Opened {url}. Vision will update; analyze the new page."

    @tool
    def click(element_id: int) -> str:
        """Click the interactable element with this numeric overlay ID."""
        is_dangerous, details = env.is_dangerous_element(element_id)
        if is_dangerous:
            return (
                "Security layer: potentially destructive click blocked via click(). "
                f"{details} Use dangerous_action with an explicit description."
            )
        return env.click_element(element_id)

    @tool
    def type_text(element_id: int, text: str) -> str:
        """Type text into the input or field with this overlay ID."""
        return env.type_text(element_id, text)

    @tool
    def press_enter() -> str:
        """Press Enter (e.g. submit search)."""
        env.page.keyboard.press("Enter")
        try:
            env.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        time.sleep(1.0)
        return "Enter pressed; page may have updated."

    @tool
    def dangerous_action(element_id: int, action_description: str) -> str:
        """ONLY for destructive actions (delete, spam, send, etc.). Requires user confirmation."""
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
        """Reserved for full-run completion (not used by the subtask executor)."""
        return f"Task finished. Report: {report}"

    return [navigate, click, type_text, press_enter, dangerous_action, finish_task]


env = BrowserEnv(headless=False)
tools = build_browser_tools(env)
