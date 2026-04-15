import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

# Импортируем наш браузер
from entry import BrowserEnv

# Инициализируем браузер (делаем глобальным, чтобы инструменты имели к нему доступ)
env = BrowserEnv(headless=False)

# ==========================================
# 1. ИНСТРУМЕНТЫ АГЕНТА (TOOLS)
# ==========================================

@tool
def get_page_state() -> str:
    """Возвращает список интерактивных элементов на странице в формате [ID] тип - 'Текст'.
    ОБЯЗАТЕЛЬНО вызывай этот инструмент после КАЖДОГО перехода или клика, чтобы увидеть новые элементы!"""
    return env.parse_page()

@tool
def navigate(url: str) -> str:
    """Переходит по указанному URL."""
    env.go_to(url)
    return f"Успешно перешли на {url}. Теперь вызови get_page_state(), чтобы осмотреться."

@tool
def click(element_id: int) -> str:
    """Кликает по элементу с указанным ID. 
    Используй для БЕЗОПАСНЫХ действий (чтение писем, навигация, открытие меню)."""
    return env.click_element(element_id)

@tool
def type_text(element_id: int, text: str) -> str:
    """Вводит текст в поле с указанным ID."""
    return env.type_text(element_id, text)

@tool
def press_enter() -> str:
    """Нажимает клавишу Enter (полезно после ввода текста)."""
    env.page.keyboard.press("Enter")
    env.page.wait_for_load_state("networkidle")
    return "Клавиша Enter нажата."

@tool
def dangerous_action(element_id: int, action_description: str) -> str:
    """ИСПОЛЬЗУЙ ТОЛЬКО ДЛЯ ДЕСТРУКТИВНЫХ ДЕЙСТВИЙ (удаление писем, спам, отправка, оплата).
    Сначала запрашивает разрешение у пользователя через терминал.
    В action_description кратко опиши, что ты хочешь сделать (например: 'Удалить спам-письмо от XXX')."""
    
    # --- SECURITY LAYER ИЗ ТРЕБОВАНИЙ ТЕСТОВОГО ---
    print(f"\n[SECURITY ALERT] Агент планирует ОПАСНОЕ ДЕЙСТВИЕ:")
    print(f"👉 Цель: {action_description}")
    user_input = input("Разрешить выполнение? [y/n]: ")
    
    if user_input.lower() == 'y':
        result = env.click_element(element_id)
        return f"Пользователь РАЗРЕШИЛ действие. Результат клика: {result}"
    else:
        return "Пользователь ЗАПРЕТИЛ действие. Не нажимай на эту кнопку, найди другой путь или пропусти письмо."

@tool
def finish_task(report: str) -> str:
    """Вызови этот инструмент, когда задача полностью выполнена.
    Передай в report итоговый отчет для пользователя."""
    return f"Задача завершена. Отчет: {report}"

tools =[get_page_state, navigate, click, type_text, press_enter, dangerous_action, finish_task]

# ==========================================
# 2. СБОРКА LANGGRAPH
# ==========================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Инициализируем LLM (gpt-4o отлично работает с Tools, но можно и claude-3-5-sonnet-20240620)

# llm = ChatOpenAI(model="gpt-4o", temperature=0)
# llm_with_tools = llm.bind_tools(tools)

llm = ChatOllama(model="qwen2.5", temperature=0)
llm_with_tools = llm.bind_tools(tools)

system_prompt = """Ты автономный AI-веб-агент. Твоя задача - управлять браузером и выполнять поручения пользователя.
ПРАВИЛА:
1. Ты НЕ видишь страницу автоматически. Чтобы понять, что на экране, ТЫ ОБЯЗАН СНАЧАЛА ВЫЗВАТЬ get_page_state().
2. Получив список элементов с их ID, ты можешь вызывать click(), type_text() и т.д.
3. После каждого действия (клик, ввод, переход) DOM-дерево меняется! ОБЯЗАТЕЛЬНО вызывай get_page_state() снова, чтобы получить новые ID элементов.
4. Если нужно удалить письмо или нажать кнопку "Спам" — ОБЯЗАТЕЛЬНО используй dangerous_action() вместо обычного click().
5. Рассуждай шаг за шагом. Не пытайся сделать всё за один вызов.
6. Когда выполнишь всю задачу, вызови finish_task(твое резюме)."""

def agent_node(state: AgentState):
    messages = state["messages"]
    # Подсовываем системный промпт, если его еще нет
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=system_prompt)] + messages
        
    response = llm_with_tools.invoke(messages)
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