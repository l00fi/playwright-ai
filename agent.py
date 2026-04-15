import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langchain_core.tools import tool
import base64

from dotenv import load_dotenv
load_dotenv()

# Импортируем наш браузер
from entry import BrowserEnv
import prompts

# Инициализируем браузер (делаем глобальным, чтобы инструменты имели к нему доступ)
env = BrowserEnv(headless=False)

# ==========================================
# 1. ИНСТРУМЕНТЫ АГЕНТА (TOOLS)
# ==========================================

@tool
def navigate(url: str) -> str:
    """Переходит по указанному URL."""
    env.go_to(url)
    # ИСПРАВЛЕНО: Убрали упоминание get_page_state
    return f"Успешно перешли на {url}. Твое зрение обновлено, анализируй новую страницу."

@tool
def click(element_id: int) -> str:
    """Кликает по элементу с указанным ID."""
    return env.click_element(element_id)

@tool
def type_text(element_id: int, text: str) -> str:
    """Вводит текст в поле с указанным ID."""
    return env.type_text(element_id, text)

@tool
def press_enter() -> str:
    """Нажимает клавишу Enter."""
    env.page.keyboard.press("Enter")
    try:
        env.page.wait_for_load_state("domcontentloaded", timeout=5000)
    except:
        pass
    time.sleep(2)
    return "Клавиша Enter нажата. Страница обновлена."

@tool
def dangerous_action(element_id: int, action_description: str) -> str:
    """ИСПОЛЬЗУЙ ТОЛЬКО ДЛЯ ДЕСТРУКТИВНЫХ ДЕЙСТВИЙ (удаление, спам, отправка).
    Запрашивает подтверждение у пользователя."""
    print(f"\n[SECURITY ALERT] Агент планирует ОПАСНОЕ ДЕЙСТВИЕ:")
    print(f"👉 Цель: {action_description}")
    user_input = input("Разрешить выполнение? [y/n]: ")
    
    if user_input.lower() == 'y':
        result = env.click_element(element_id)
        return f"Пользователь РАЗРЕШИЛ действие. Результат: {result}"
    else:
        return "Пользователь ОТМЕНИЛ действие. Тебе нужно найти другой путь или пропустить это."

@tool
def finish_task(report: str) -> str:
    """Вызывается, когда задача полностью выполнена."""
    return f"Задача завершена. Отчет: {report}"

tools =[navigate, click, type_text, press_enter, dangerous_action, finish_task]

# ==========================================
# 2. СБОРКА LANGGRAPH
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Инициализируем LLM (gpt-4o отлично работает с Tools, но можно и claude-3-5-sonnet-20240620)

llm = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    # 1. Получаем свежий скриншот
    screenshot_path, text_metadata = env.get_visual_state()
    
    if not screenshot_path:
        return {"messages": [AIMessage(content="Я не вижу страницу. Попробую еще раз.")]}

    with open(screenshot_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # 2. Берем историю (без старых картинок!)
    messages = state["messages"]
    
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=prompts.SYSTEM_PROMPT)] + messages

    # 3. Формируем ОДИН HumanMessage с текущим зрением
    # Мы не добавляем его в state навсегда, а используем только для текущего вызова invoke
    current_perception = HumanMessage(content=[
        {
            "type": "text", 
            "text": f"CURRENT PAGE CONTENT:\n{text_metadata}\n\nAnalyze the screenshot and text. What is your single next step to reach the goal?"
        },
        {
            "type": "image_url", 
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        }
    ])

    # Вызов модели: история + свежий взгляд
    response = llm_with_tools.invoke(messages + [current_perception])
    
    # ВАЖНО: в State возвращаем только текстовый ответ модели, чтобы не раздувать граф
    return {"messages": [response]}

def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages =[]
    
    for tool_call in last_message.tool_calls:
        # Логируем действия агента в терминал (как просили в ТЗ!)
        print(f"\n🤖 [Агент вызывает инструмент] -> {tool_call['name']}")
        print(f"   Аргументы: {tool_call['args']}")
        
        # Выполняем инструмент
        action = next(t for t in tools if t.name == tool_call['name'])
        result = action.invoke(tool_call['args'])
        
        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call['id']))
        
    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # Если модель не вызвала инструмент или вызвала finish_task — завершаем граф
    if not last_message.tool_calls:
        return "end"
    if any(tc['name'] == 'finish_task' for tc in last_message.tool_calls):
        return "end"
    return "continue"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"continue": "tools", "end": END})
workflow.add_edge("tools", "agent")

app = workflow.compile()

# ==========================================
# 3. ЗАПУСК ПРИЛОЖЕНИЯ
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("🚀 Автономный Web-Агент запущен!")
    print("="*50)

    print("🌐 Открываем стартовую страницу...")
    env.go_to("https://ya.ru")
    
    user_task = input("\n📝 Введите задачу для агента: ")
    
    inputs = {"messages":[HumanMessage(content=user_task)]}
    
    # recursion_limit=50 ограничивает агента от бесконечного зацикливания
    for output in app.stream(inputs, {"recursion_limit": 50}):
        if "agent" in output:
            ai_msg = output["agent"]["messages"][-1]
            # Выводим мысли агента (если модель их генерирует перед вызовом тулзов)
            if ai_msg.content:
                print(f"\n🧠 [Мысли агента]: {ai_msg.content}")
                
    print("\n✅ РАБОТА ЗАВЕРШЕНА. Закрываю браузер...")
    env.close()