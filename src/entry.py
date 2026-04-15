import json
import os
import shutil
import time
from datetime import datetime
from playwright.sync_api import sync_playwright


def _env_int(name: str, default: int, min_value: int = 0) -> int:
    raw = os.getenv(name, "")
    try:
        return max(int(raw), min_value)
    except Exception:
        return default


class BrowserEnv:
    def __init__(self, headless: bool = False, user_data_dir: str = "./browser_data"):
        self.playwright = sync_playwright().start()
        self.context = self.playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=headless,
            viewport={"width": 1280, "height": 800},
            args=["--disable-blink-features=AutomationControlled"],
        )
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        self.run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.snapshot_dir = os.path.join("artifacts", "snapshots", self.run_id)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        self.snapshot_counter = 0
        self.tab_redirect_events: list[dict] = []
        self.context.on("page", self._on_new_page)
        self._ensure_single_tab()

    def _on_new_page(self, new_page):
        if new_page == self.page:
            return
        try:
            new_page.wait_for_load_state("domcontentloaded", timeout=2500)
        except Exception:
            pass

        try:
            target_url = new_page.url
            if target_url and target_url != "about:blank":
                self.tab_redirect_events.append(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "from_new_tab_url": target_url,
                    }
                )
                self.page.goto(target_url, wait_until="domcontentloaded", timeout=20000)
        except Exception:
            pass

        try:
            new_page.close()
        except Exception:
            pass
        self._ensure_single_tab()

    def _ensure_single_tab(self):
        pages = list(self.context.pages)
        if self.page not in pages and pages:
            self.page = pages[0]
        for p in list(self.context.pages):
            if p != self.page:
                try:
                    p.close()
                except Exception:
                    pass

    def _clear_overlays(self):
        script = """
        () => {
            document.querySelectorAll('.ai-label').forEach(el => el.remove());
            document.querySelectorAll('[data-ai-highlight]').forEach(el => {
                el.style.outline = '';
                el.removeAttribute('data-ai-highlight');
                el.removeAttribute('data-ai-id');
            });
        }
        """
        try:
            self.page.evaluate(script)
        except Exception:
            pass

    def _clear_floating_labels_only(self):
        """Remove only SoM number badges; keep data-ai-id so click_element can still target the element."""
        try:
            self.page.evaluate("() => { document.querySelectorAll('.ai-label').forEach(el => el.remove()); }")
        except Exception:
            pass

    def go_to(self, url: str):
        self._ensure_single_tab()
        self._clear_overlays()
        self.page.goto(url, wait_until="load", timeout=25000)

    def _wait_until_page_ready_for_screenshot(self) -> tuple[bool, bool]:
        """
        Returns (ok_for_snapshot, degraded).
        degraded=True means we skipped strict SPA idle wait (busy/long pages) but DOM is usable.
        """
        mutation_timeout = _env_int("AGENT_SNAPSHOT_MUTATION_WAIT_MS", 6000, min_value=500)
        idle_ms = _env_int("AGENT_SNAPSHOT_MUTATION_IDLE_MS", 500, min_value=100)
        load_timeout = _env_int("AGENT_SNAPSHOT_LOAD_WAIT_MS", 8000, min_value=1000)

        try:
            self.page.wait_for_load_state("load", timeout=min(load_timeout, 10000))
        except Exception:
            try:
                self.page.wait_for_load_state("domcontentloaded", timeout=5000)
            except Exception:
                return False, False

        idle_script = f"""() => {{
            if (document.readyState !== "complete") return false;
            if (document.fonts && document.fonts.status !== "loaded") return false;
            if (!window.__aiObserverInstalled) {{
                window.__aiObserverInstalled = true;
                window.__aiLastMutationTs = Date.now();
                const observer = new MutationObserver(() => {{ window.__aiLastMutationTs = Date.now(); }});
                observer.observe(document.documentElement || document.body, {{
                    subtree: true, childList: true, attributes: true, characterData: false
                }});
            }}
            const idleMs = Date.now() - (window.__aiLastMutationTs || 0);
            return idleMs >= {idle_ms};
        }}"""

        try:
            self.page.wait_for_function(idle_script, timeout=mutation_timeout)
            return True, False
        except Exception:
            pass

        # Fallback: never block screenshots on perpetually-mutating SPAs (Habr, mail, etc.)
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=3000)
        except Exception:
            pass
        try:
            self.page.evaluate(
                "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
            )
            self.page.wait_for_timeout(_env_int("AGENT_SNAPSHOT_FALLBACK_SETTLE_MS", 350, min_value=50))
        except Exception:
            pass
        return True, True

    def _draw_labels(self) -> int:
        script = """
        () => {
            document.querySelectorAll('.ai-label').forEach(el => el.remove());
            document.querySelectorAll('[data-ai-highlight]').forEach(el => {
                el.style.outline = '';
                el.removeAttribute('data-ai-highlight');
                el.removeAttribute('data-ai-id');
            });

            const viewportH = window.innerHeight || 800;
            const viewportW = window.innerWidth || 1280;
            const selector = 'a, button, input, textarea, select, [role="button"], [role="link"], [role="tab"], [onclick], [contenteditable="true"], .button';
            const elements = document.querySelectorAll(selector);
            let aiId = 1;
            let labeledCount = 0;
            const maxLabels = 180;

            elements.forEach(el => {
                if (labeledCount >= maxLabels) return;
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                const inViewport = rect.bottom > 0 && rect.right > 0 && rect.top < viewportH && rect.left < viewportW;
                if (
                    rect.width > 5 &&
                    rect.height > 5 &&
                    inViewport &&
                    style.visibility !== 'hidden' &&
                    style.display !== 'none' &&
                    style.opacity !== '0' &&
                    style.pointerEvents !== 'none'
                ) {
                    el.style.outline = '1px solid #ff2d2d';
                    el.style.outlineOffset = '0px';
                    el.setAttribute('data-ai-highlight', 'true');
                    el.setAttribute('data-ai-id', aiId);

                    const label = document.createElement('div');
                    label.className = 'ai-label';
                    label.innerText = aiId;
                    label.style.position = 'fixed';
                    label.style.top = Math.max(0, Math.min(viewportH - 18, rect.top)) + 'px';
                    label.style.left = Math.max(0, Math.min(viewportW - 22, rect.left)) + 'px';
                    label.style.backgroundColor = 'rgba(255, 0, 0, 0.92)';
                    label.style.color = '#ffffff';
                    label.style.fontWeight = 'bold';
                    label.style.fontSize = '11px';
                    label.style.lineHeight = '1.1';
                    label.style.padding = '0px 3px';
                    label.style.border = '1px solid white';
                    label.style.borderRadius = '3px';
                    label.style.zIndex = '2147483647';
                    label.style.pointerEvents = 'none';
                    document.body.appendChild(label);

                    aiId += 1;
                    labeledCount += 1;
                }
            });
            return aiId - 1;
        }
        """
        return self.page.evaluate(script)

    def parse_page(self) -> str:
        js_script = """
        () => {
            const interactable = [];
            const marked = Array.from(document.querySelectorAll('[data-ai-id]'))
                .sort((a, b) => Number(a.getAttribute('data-ai-id')) - Number(b.getAttribute('data-ai-id')));

            for (const el of marked) {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                if (rect.width <= 0 || rect.height <= 0 || style.visibility === 'hidden' || style.display === 'none') continue;
                const id = el.getAttribute('data-ai-id');
                let text = el.innerText || el.placeholder || el.value || el.getAttribute('aria-label') || el.getAttribute('title') || '';
                text = text.trim().substring(0, 80).replace(/\\n/g, ' ');
                if (!text) text = 'No text';
                interactable.push(`[ID: ${id}] ${el.tagName.toLowerCase()} - '${text}'`);
            }

            const contentSelectors = [
                'main article h1', 'main article h2', 'main article h3', 'main article p',
                'main p', 'article p', 'article h1', 'article h2', '[role="article"] h1',
                '[role="article"] p', '.tm-title__link', '.tm-article-snippet__title',
                '[data-test-id="articleTitle"]', '.post_title', '.post__title',
                'h1', 'h2', 'h3', 'li'
            ];
            const contentCandidates = Array.from(
                document.querySelectorAll(contentSelectors.join(', '))
            );
            const seen = new Set();
            const visibleTextChunks = [];
            for (const el of contentCandidates) {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                if (rect.width <= 0 || rect.height <= 0) continue;
                if (style.visibility === 'hidden' || style.display === 'none') continue;
                if (rect.bottom < 0 || rect.top > window.innerHeight) continue;
                let text = (el.innerText || '').replace(/\\s+/g, ' ').trim();
                if (!text || text.length < 12) continue;
                if (text.length > 280) text = text.slice(0, 280) + '...';
                if (seen.has(text)) continue;
                seen.add(text);
                visibleTextChunks.push(text);
                if (visibleTextChunks.length >= 40) break;
            }

            const interactiveText = interactable.join('\\n') || 'No interactive elements detected.';
            const visibleText = visibleTextChunks.join('\\n') || 'No substantial visible text extracted.';
            return "INTERACTIVE ELEMENTS:\\n" + interactiveText + "\\n\\nVISIBLE PAGE TEXT:\\n" + visibleText;
        }
        """
        return self.page.evaluate(js_script)

    def get_visual_state(self):
        ready, degraded = self._wait_until_page_ready_for_screenshot()
        if not ready:
            return None, "Ошибка визуализации: страница не достигла даже базовой готовности (DOM)."

        try:
            labels_count = 0
            for _ in range(2):
                labels_count = self._draw_labels()
                self.page.evaluate(
                    "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
                )
                self.page.wait_for_timeout(120)
                if labels_count > 0:
                    break

            # Second stability pass: on failure, keep degraded snapshot instead of aborting.
            ready2, deg2 = self._wait_until_page_ready_for_screenshot()
            degraded = degraded or deg2 or not ready2

            screenshot_path = "screenshot.png"
            self.page.screenshot(path=screenshot_path)
            text_state = self.parse_page()
            prefix = ""
            if degraded:
                prefix = "[SNAPSHOT_DEGRADED] Page may still be animating; trust visible text/screenshot with caution.\\n"
            if labels_count == 0:
                text_state = f"{prefix}[WARNING] No visible interactive elements were marked.\\n{text_state}"
            else:
                text_state = prefix + text_state
            self._persist_snapshot_artifacts(
                screenshot_path=screenshot_path,
                text_state=text_state,
                labels_count=labels_count,
            )
            return screenshot_path, text_state
        except Exception as e:
            return None, f"Ошибка визуализации: {e}"

    def _persist_snapshot_artifacts(self, screenshot_path: str, text_state: str, labels_count: int):
        """
        Сохраняет каждый snapshot, который передается модели:
        - png (точная копия текущего screenshot)
        - json с метаданными и текстовым состоянием
        """
        self.snapshot_counter += 1
        shot_name = f"step-{self.snapshot_counter:04d}.png"
        meta_name = f"step-{self.snapshot_counter:04d}.json"
        shot_path = os.path.join(self.snapshot_dir, shot_name)
        meta_path = os.path.join(self.snapshot_dir, meta_name)

        try:
            shutil.copyfile(screenshot_path, shot_path)
        except Exception:
            # fallback: пробуем сохранить повторно напрямую
            try:
                self.page.screenshot(path=shot_path)
            except Exception:
                pass

        pages_info = []
        for idx, p in enumerate(list(self.context.pages)):
            try:
                pages_info.append({"idx": idx, "url": p.url})
            except Exception:
                pages_info.append({"idx": idx, "url": "<unavailable>"})

        try:
            current_url = self.page.url
        except Exception:
            current_url = "<unavailable>"
        try:
            current_title = self.page.title()
        except Exception:
            current_title = "<unavailable>"

        metadata = {
            "snapshot_step": self.snapshot_counter,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_id": self.run_id,
            "screenshot_path_runtime": screenshot_path,
            "screenshot_path_saved": shot_path,
            "labels_count": labels_count,
            "current_url": current_url,
            "current_title": current_title,
            "open_tabs_count": len(list(self.context.pages)),
            "open_tabs": pages_info,
            "recent_tab_redirect_events": self.tab_redirect_events[-10:],
            "text_state": text_state,
        }

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def is_dangerous_element(self, element_id: int):
        selector = f'[data-ai-id="{element_id}"]'
        js_script = """
        (sel) => {
            const el = document.querySelector(sel);
            if (!el) return { exists: false, dangerous: false, text: "" };
            const raw = (el.innerText || el.value || el.getAttribute('aria-label') || el.getAttribute('title') || "").toLowerCase().trim();
            // "spam/спам" alone is usually a folder nav label — do not block. Block destructive phrases instead.
            const folderNav = /^(спам|spam|junk)(\\s|\\(|$)/i.test(raw) && raw.length < 48;
            if (folderNav) return { exists: true, dangerous: false, text: raw };
            const dangerPhrases = [
                "delete", "remove", "trash", "unsubscribe", "submit",
                "report spam", "move to spam", "mark as spam",
                "удал", "отправить", "очист", "подтверд", "подпис",
                "переместить в спам", "пометить как спам", "в спам"
            ];
            const dangerous = dangerPhrases.some((p) => raw.includes(p));
            return { exists: true, dangerous, text: raw };
        }
        """
        try:
            result = self.page.evaluate(js_script, selector)
            if not result.get("exists", False):
                return False, "Элемент не найден"
            if result.get("dangerous", False):
                return True, f"Обнаружен рискованный текст элемента: '{result.get('text', '')[:120]}'"
            return False, ""
        except Exception as e:
            return False, f"Не удалось проверить элемент: {e}"

    def _click_at_box_center(self, x: float, y: float, w: float, h: float) -> None:
        cx = x + max(w / 2, 0.5)
        cy = y + max(h / 2, 0.5)
        self.page.mouse.click(cx, cy)

    def click_element(self, element_id: int):
        self._ensure_single_tab()
        self._clear_floating_labels_only()
        selector = f'[data-ai-id="{element_id}"]'
        loc = self.page.locator(selector)
        try:
            try:
                loc.scroll_into_view_if_needed(timeout=2000)
            except Exception:
                pass
            loc.click(timeout=5000)
        except Exception as first_err:
            note = str(first_err)
            try:
                loc.click(timeout=4000, force=True)
                self._after_click_success()
                return f"Успешно кликнул по ID {element_id} (force=True; первый клик: {note[:200]})"
            except Exception:
                pass
            box = None
            try:
                box = loc.bounding_box(timeout=2000)
            except Exception:
                box = None
            if box and box.get("width", 0) > 0 and box.get("height", 0) > 0:
                try:
                    self._click_at_box_center(box["x"], box["y"], box["width"], box["height"])
                    self._after_click_success()
                    return (
                        f"Успешно кликнул по ID {element_id} (page.mouse по центру bbox; "
                        f"обычный клик: {note[:180]})"
                    )
                except Exception as mouse_err:
                    pass
            try:
                rect = self.page.evaluate(
                    """
                    (sel) => {
                        const el = document.querySelector(sel);
                        if (!el) return null;
                        const r = el.getBoundingClientRect();
                        if (r.width < 1 || r.height < 1) return null;
                        return { x: r.left, y: r.top, w: r.width, h: r.height };
                    }
                    """,
                    selector,
                )
                if rect:
                    self._click_at_box_center(rect["x"], rect["y"], rect["w"], rect["h"])
                    self._after_click_success()
                    return (
                        f"Успешно кликнул по ID {element_id} (page.mouse по getBoundingClientRect; "
                        f"обычный клик: {note[:180]})"
                    )
            except Exception:
                pass
            return f"Ошибка клика по ID {element_id}: {note}"

        self._after_click_success()
        return f"Успешно кликнул по ID {element_id}"

    def _after_click_success(self):
        self._ensure_single_tab()
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=5000)
        except Exception:
            pass
        time.sleep(1.0)

    def type_text(self, element_id: int, text: str):
        self._clear_floating_labels_only()
        selector = f'[data-ai-id="{element_id}"]'
        try:
            loc = self.page.locator(selector)
            try:
                loc.scroll_into_view_if_needed(timeout=2000)
            except Exception:
                pass
            loc.fill(text)
            return f"Успешно ввел текст в ID {element_id}"
        except Exception as e:
            return f"Ошибка ввода: {e}"

    def close(self):
        try:
            if getattr(self, "context", None) is not None:
                self.context.close()
        except Exception:
            pass
        try:
            if getattr(self, "playwright", None) is not None:
                self.playwright.stop()
        except Exception:
            pass


def _run_manual_session_cli() -> None:
    """
    Отдельный запуск браузера с тем же persistent-профилем, что использует агент (./browser_data).
    Удобно залогиниться и пройти 2FA вручную, затем закрыть окно и запустить python main.py.
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Открыть Chromium с профилем агента, перейти на URL, дать время войти вручную. "
            "После Enter браузер закроется — cookies останутся в user-data-dir."
        )
    )
    parser.add_argument(
        "url",
        nargs="?",
        default=os.getenv("BROWSER_PREP_URL", "https://ya.ru"),
        help="Стартовый адрес (по умолчанию ya.ru или переменная BROWSER_PREP_URL)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Без окна (для логина не подходит)",
    )
    parser.add_argument(
        "--user-data-dir",
        default="./browser_data",
        help="Каталог профиля; должен совпадать с BrowserEnv в agent_tools (по умолчанию ./browser_data).",
    )
    args = parser.parse_args()
    udir = os.path.abspath(args.user_data_dir)
    url = (args.url or "").strip() or "https://ya.ru"

    print(f"Профиль: {udir}", file=sys.stderr)
    print(f"Стартовый URL: {url}", file=sys.stderr)

    env = BrowserEnv(headless=args.headless, user_data_dir=udir)
    try:
        env.go_to(url)
        print(
            "\nОкно браузера открыто. Выполните вход / 2FA вручную при необходимости.\n"
            "Когда сессия будет готова — нажмите Enter здесь, чтобы закрыть браузер и сохранить профиль.\n"
            "Затем запустите агента: python main.py\n",
            file=sys.stderr,
        )
        try:
            input("Enter — закрыть браузер… ")
        except EOFError:
            pass
    finally:
        try:
            env.close()
        except Exception as e:
            print(f"Примечание при закрытии браузера (можно игнорировать, если окно уже закрывали вручную): {e}", file=sys.stderr)
        print("Готово.", file=sys.stderr)


if __name__ == "__main__":
    _run_manual_session_cli()
