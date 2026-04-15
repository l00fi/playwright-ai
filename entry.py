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
        """Переход по URL с мягким ожиданием"""
        try:
            # Ждем просто загрузки основного контента (load)
            self.page.goto(url, wait_until="load", timeout=20000)
        except Exception as e:
            print(f"⚠️ Загрузка {url} заняла слишком много времени, но продолжаем...")

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

    def get_visual_state(self):
        """
        Улучшенная отрисовка меток и захват скриншота.
        """
        # Сначала даем странице "продышаться" после предыдущего действия
        time.sleep(2) 
        
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
        }
        """
        try:
            self.page.evaluate(draw_script)
            # Короткая пауза, чтобы браузер успел отрисовать красные квадраты
            time.sleep(0.8) 
            
            screenshot_path = "screenshot.png"
            self.page.screenshot(path=screenshot_path)
            
            # Получаем текстовое описание для модели
            text_state = self.parse_page() 
            return screenshot_path, text_state
        except Exception as e:
            return None, f"Ошибка визуализации: {str(e)}"

    def click_element(self, element_id: int):
        """Кликает по элементу и не падает от фоновых запросов"""
        selector = f'[data-ai-id="{element_id}"]'
        try:
            self.page.locator(selector).click(timeout=5000)
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