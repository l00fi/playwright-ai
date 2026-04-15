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

    def go_to(self, url: str):
        """Переход по URL"""
        self.page.goto(url)
        self.page.wait_for_load_state("networkidle")

    def parse_page(self):
        """
        МАГИЯ ЗДЕСЬ: JS-скрипт, который находит все интерактивные элементы,
        присваивает им кастомный атрибут data-ai-id и возвращает компактный список.
        Это решает проблему токенов и избавляет от хардкода селекторов!
        """
        js_script = """
        () => {
            let ai_id = 1;
            let interactable =[];
            // Ищем ссылки, кнопки, инпуты и элементы с ролью button
            let elements = document.querySelectorAll('a, button, input, textarea, [role="button"]');
            
            elements.forEach(el => {
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                
                // Берем только видимые элементы
                if(rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none') {
                    el.setAttribute('data-ai-id', ai_id);
                    
                    // Пытаемся достать осмысленный текст элемента
                    let text = el.innerText || el.placeholder || el.value || el.getAttribute('aria-label') || 'No text';
                    text = text.trim().substring(0, 60).replace(/\\n/g, ' '); // Режем длинные тексты
                    
                    if (text !== 'No text' && text !== '') {
                        interactable.push(`[ID: ${ai_id}] ${el.tagName.toLowerCase()} - '${text}'`);
                        ai_id++;
                    }
                }
            });
            return interactable.join('\\n');
        }
        """
        # Выполняем скрипт на странице
        compact_dom = self.page.evaluate(js_script)
        return compact_dom

    def click_element(self, element_id: int):
        """Кликает по элементу, используя наш сгенерированный ID"""
        selector = f'[data-ai-id="{element_id}"]'
        try:
            self.page.locator(selector).click(timeout=3000)
            self.page.wait_for_load_state("networkidle")
            time.sleep(1) # Небольшая пауза для рендеринга JS
            return f"Успешно кликнул по ID {element_id}"
        except Exception as e:
            return f"Ошибка клика: {str(e)}"

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
    
    print("Открываем Яндекс...")
    env.go_to("https://ya.ru")
    
    print("Собираем элементы...")
    state = env.parse_page()
    
    print("\n--- ТО, ЧТО УВИДИТ LLM ---")
    print(state)
    print("--------------------------\n")
    
    input("Нажми Enter, чтобы закрыть браузер...")
    env.close()