from playwright.sync_api import sync_playwright
import time

class BrowserEnv:
    def __init__(self, headless=False, user_data_dir="./browser_data"):
        self.playwright = sync_playwright().start()
        # Persistent context позволяет сохранять куки и сессии (например, логин в Яндексе)
        self.context = self.playwright.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=headless,
            viewport={"width": 1280, "height": 800},
            args=["--disable-blink-features=AutomationControlled"] # Немного прячемся от антифрода
        )
        self.page = self.context.pages[0] if self.context.pages else self.context.new_page()
        self._install_single_tab_guards()
        self._ensure_single_tab()

    def _install_single_tab_guards(self):
        """Гарантирует работу только в одной вкладке."""
        self.context.on("page", self._on_new_page)

    def _on_new_page(self, new_page):
        """Перенаправляет неожиданные новые вкладки в основную и закрывает их."""
        if new_page == self.page:
            return

        try:
            new_page.wait_for_load_state("domcontentloaded", timeout=2500)
        except Exception:
            pass

        target_url = ""
        try:
            target_url = new_page.url
        except Exception:
            pass

        # about:blank не несет полезной навигации — просто закрываем.
        if target_url and target_url != "about:blank":
            try:
                self.page.goto(target_url, wait_until="domcontentloaded", timeout=20000)
            except Exception:
                pass

        try:
            new_page.close()
        except Exception:
            pass

        self._ensure_single_tab()

    def _ensure_single_tab(self):
        """Оставляет только self.page активной вкладкой в контексте."""
        pages = list(self.context.pages)
        if self.page not in pages and pages:
            self.page = pages[0]
        for p in list(self.context.pages):
            if p != self.page:
                try:
                    p.close()
                except Exception:
                    pass

    def go_to(self, url: str):
        """Переход по URL с мягким ожиданием"""
        self._ensure_single_tab()
        try:
            # Ждем просто загрузки основного контента (load)
            self.page.goto(url, wait_until="load", timeout=20000)
        except Exception as e:
            print(f"⚠️ Загрузка {url} заняла слишком много времени, но продолжаем...")

    def parse_page(self):
        """
        Собираем компактное текстовое состояние по уже размеченным data-ai-id.
        Важно не перезаписывать ID после отрисовки, чтобы текст и скриншот совпадали.
        """
        js_script = """
        () => {
            const interactable = [];
            const marked = Array.from(document.querySelectorAll('[data-ai-id]'))
                .sort((a, b) => Number(a.getAttribute('data-ai-id')) - Number(b.getAttribute('data-ai-id')));

            for (const el of marked) {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                if (rect.width <= 0 || rect.height <= 0 || style.visibility === 'hidden' || style.display === 'none') {
                    continue;
                }

                const id = el.getAttribute('data-ai-id');
                let text = el.innerText || el.placeholder || el.value || el.getAttribute('aria-label') || el.getAttribute('title') || '';
                text = text.trim().substring(0, 80).replace(/\\n/g, ' ');
                if (!text) text = 'No text';
                interactable.push(`[ID: ${id}] ${el.tagName.toLowerCase()} - '${text}'`);
            }

            return interactable.join('\\n');
        }
        """
        # Выполняем скрипт на странице
        compact_dom = self.page.evaluate(js_script)
        return compact_dom

    def get_visual_state(self):
        """
        Улучшенная отрисовка меток и захват скриншота.
        """
        # Короткое "успокоение" DOM после прошлого действия.
        self.page.wait_for_timeout(300)
        try:
            self.page.wait_for_load_state("domcontentloaded", timeout=2500)
        except Exception:
            pass

        draw_script = """
        () => {
            // 1. Удаляем все старые метки и обводки
            document.querySelectorAll('.ai-label').forEach(el => el.remove());
            document.querySelectorAll('[data-ai-highlight]').forEach(el => {
                el.style.outline = '';
                el.removeAttribute('data-ai-highlight');
            });

            let ai_id = 1;
            // Ищем все потенциально кликабельные элементы
            const elements = document.querySelectorAll('a, button, input, textarea, [role="button"], [onclick], .button, [role="link"]');
            
            elements.forEach(el => {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                
                // Проверяем видимость
                if (rect.width > 5 && rect.height > 5 && style.visibility !== 'hidden' && style.display !== 'none') {
                    
                    // Подсвечиваем сам элемент обводкой
                    el.style.outline = '2px dashed red';
                    el.setAttribute('data-ai-highlight', 'true');
                    el.setAttribute('data-ai-id', ai_id);

                    // Создаем плашку с номером
                    const label = document.createElement('div');
                    label.className = 'ai-label';
                    label.innerText = ai_id;
                    label.style.position = 'absolute';
                    label.style.top = (rect.top + window.scrollY) + 'px';
                    label.style.left = (rect.left + window.scrollX) + 'px';
                    label.style.backgroundColor = '#ff0000';
                    label.style.color = '#ffffff';
                    label.style.fontWeight = 'bold';
                    label.style.fontSize = '14px';
                    label.style.padding = '1px 4px';
                    label.style.border = '1px solid white';
                    label.style.borderRadius = '4px';
                    label.style.zIndex = '2147483647'; // Максимальный z-index
                    label.style.pointerEvents = 'none';
                    
                    document.body.appendChild(label);
                    ai_id++;
                }
            });
            return ai_id - 1;
        }
        """
        try:
            self.page.evaluate(draw_script)
            # Даем браузеру 2 кадра на финальную отрисовку оверлеев.
            self.page.evaluate(
                "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
            )
            
            screenshot_path = "screenshot.png"
            self.page.screenshot(path=screenshot_path)
            
            # Получаем текстовое описание для модели
            text_state = self.parse_page() 
            return screenshot_path, text_state
        except Exception as e:
            return None, f"Ошибка визуализации: {str(e)}"

    def is_dangerous_element(self, element_id: int):
        """Пытается распознать потенциально деструктивный клик по тексту элемента."""
        selector = f'[data-ai-id="{element_id}"]'
        js_script = """
        (sel) => {
            const el = document.querySelector(sel);
            if (!el) return { exists: false, dangerous: false, text: "" };
            const raw = (
                el.innerText ||
                el.value ||
                el.getAttribute('aria-label') ||
                el.getAttribute('title') ||
                ""
            ).toLowerCase().trim();
            const dangerWords = [
                "delete", "remove", "trash", "spam", "send", "submit", "unsubscribe",
                "удал", "спам", "отправ", "очист", "подтверд", "подпис"
            ];
            const dangerous = dangerWords.some(word => raw.includes(word));
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
            return False, f"Не удалось проверить элемент: {str(e)}"

    def click_element(self, element_id: int):
        """Кликает по элементу и не падает от фоновых запросов"""
        self._ensure_single_tab()
        selector = f'[data-ai-id="{element_id}"]'
        try:
            self.page.locator(selector).click(timeout=5000)
            self._ensure_single_tab()
            # Вместо networkidle ждем просто загрузки DOM
            self.page.wait_for_load_state("domcontentloaded", timeout=5000)
            time.sleep(2) # Даем 2 секунды на случайную анимацию
            return f"Успешно кликнул по ID {element_id}"
        except Exception as e:
            return f"Ошибка клика по ID {element_id}: {str(e)}"

    def type_text(self, element_id: int, text: str):
        """Вводит текст в элемент"""
        selector = f'[data-ai-id="{element_id}"]'
        try:
            self.page.locator(selector).fill(text)
            return f"Успешно ввел текст в ID {element_id}"
        except Exception as e:
            return f"Ошибка ввода: {str(e)}"

    def close(self):
        self.context.close()
        self.playwright.stop()

# --- Блок для проверки ---
if __name__ == "__main__":
    env = BrowserEnv(headless=False) # headless=False чтобы мы видели браузер
    
    addres_str = "https://ya.ru" 
    print(f"Открываем {addres_str}...")
    env.go_to(addres_str)
    
    print("Собираем элементы...")
    screenshot_path, text_metadata = env.get_visual_state()
    
    print(screenshot_path)
    print(text_metadata)
    
    input("Нажми Enter, чтобы закрыть браузер...")
    env.close()