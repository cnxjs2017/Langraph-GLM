import operator
import requests
import uuid
from typing import TypedDict, Union, List, Annotated

from langchain.agents import create_openai_tools_agent
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_community.chat_models import ChatZhipuAI
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# API密钥
api_key = ""

# 初始化ChatZhipuAI模型
chat_zhipu = ChatZhipuAI(
    temperature=0.8,
    api_key=api_key,
    model="glm-4-flash"
)


# 定义联网搜索工具
@tool("web_search")
def web_search(query: str) -> str:
    """
    调用联网搜索工具，返回搜索结果。
    """
    print("query:", query)
    tool = "web-search-pro"
    url = "https://open.bigmodel.cn/api/paas/v4/tools"
    request_id = str(uuid.uuid4())
    data = {
        "request_id": request_id,
        "tool": tool,
        "stream": False,
        "messages": [{"role": "user", "content": query}]
    }
    headers = {'Authorization': api_key}
    resp = requests.post(url, json=data, headers=headers, timeout=300)
    print("联网搜索工具调用结果:", resp.status_code)  # 打印工具调用状态
    if resp.status_code == 200:
        result = \
            resp.json().get("choices", [{}])[0].get("message", {}).get("tool_calls", [{}])[1].get("search_result",
                                                                                                  [{}])[
                0].get("content", "未找到相关信息")
        print("工具返回结果:", result)
        return result
    else:
        return f"联网搜索失败，状态码：{resp.status_code}"


# 工具列表
tools = [web_search]

# 创建工具节点
tool_node = ToolNode(tools)

# 定义 Agent 的提示词
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手。当你无法回答问题时，请调用工具来获取信息。"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 创建 Agent
agent_runnable = create_openai_tools_agent(chat_zhipu, tools, agent_prompt)

# 定义美化 Agent 的提示词
beautify_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个文笔优化助手，负责对文本进行润色和美化，使其更加流畅和优雅。"),
    ("user", "请对以下文本进行文笔优化：\n{text}"),
])

# 创建美化 Agent
beautify_agent = beautify_prompt | chat_zhipu


# 定义状态字典
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]


# 定义节点函数
def run_agent(data: AgentState) -> dict:
    print("\n--- 进入 agent 节点 ---")
    print("当前状态:", data)
    agent_outcome = agent_runnable.invoke(data)
    # 检查 agent_outcome 是否是列表
    if isinstance(agent_outcome, list):
        # 假设我们只需要第一个元素
        agent_outcome = agent_outcome[0]

    print("Agent 输出结果:", agent_outcome)
    return {"agent_outcome": agent_outcome}


def execute_tools(data: AgentState) -> dict:
    print("\n--- 进入 action 节点 ---")
    print("当前状态:", data)
    agent_action = data["agent_outcome"]

    # 检查 agent_action 是否是列表
    if isinstance(agent_action, list):
        # 假设我们只需要第一个元素
        agent_action = agent_action[0]

    print("Agent 动作:", agent_action)

    if isinstance(agent_action, AgentAction):
        tool_name = agent_action.tool
        tool_input = agent_action.tool_input["query"]
        print(f"调用工具: {tool_name}, 输入: {tool_input}")

        # 生成唯一的 tool_call_id
        tool_call_id = str(uuid.uuid4())

        # 构造符合 ToolNode 要求的输入格式
        tool_input_dict = {
            "messages": [
                {"role": "user", "content": tool_input},  # 用户消息
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": tool_call_id,  # 添加唯一的 id
                            "name": tool_name,
                            "args": {"query": tool_input},
                        }
                    ],
                ),  # AI 消息
            ],
            "tool": tool_name,
            "tool_input": tool_input,
        }

        output = tool_node.invoke(tool_input_dict)
        print("工具执行结果:", output)
    else:
        output = "Unknown action"
    return {"intermediate_steps": data["intermediate_steps"] + [(agent_action, str(output))]}


# 定义美化节点函数
def beautify_output(data: AgentState) -> dict:
    print("\n--- 进入 beautify 节点 ---")
    print("当前状态:", data)

    # 获取原始输出
    if isinstance(data["agent_outcome"], AgentFinish):
        original_output = data["agent_outcome"].return_values["output"]
    else:
        original_output = str(data["agent_outcome"])

    print("原始输出:", original_output)

    # 调用美化 Agent
    beautified_output = beautify_agent.invoke({"text": original_output})
    print("美化后的输出:", beautified_output.content)

    # 更新状态
    return {
        "agent_outcome": AgentFinish(
            return_values={"output": beautified_output.content},
            log="文本美化完成",
        )
    }


# 定义条件判断函数
def should_continue(data: AgentState) -> str:
    print("\n--- 检查是否继续 ---")
    agent_outcome = data["agent_outcome"]
    print("agent_outcome:", agent_outcome)
    if isinstance(agent_outcome, list):
        action = agent_outcome[0]
    else:
        action = agent_outcome
    # 检查是否是 AgentFinish 类型
    if isinstance(action, AgentFinish):
        print("Agent 完成，进入美化节点")
        return "beautify"  # 进入 beautify 节点

    # 检查是否是 AgentAction 类型
    elif isinstance(action, ToolAgentAction):
        print(f"Agent 需要调用工具: {action.tool}, 输入: {action.tool_input}")
        return "continue"

    # 默认情况：未知的 Agent 输出，结束执行
    else:
        print("未知的 Agent 输出，结束执行")
        return "end"


# 创建和配置图
workflow = StateGraph(AgentState)
workflow.add_node("agent", run_agent)
workflow.add_node("action", execute_tools)
workflow.add_node("beautify", beautify_output)  # 新增 beautify 节点
workflow.set_entry_point("agent")

# 更新条件边
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "beautify": "beautify",  # 进入 beautify 节点
        "end": END,
    },
)

workflow.add_edge("action", "agent")
workflow.add_edge("beautify", END)  # beautify 节点后结束流程
app = workflow.compile()

# 运行图并获取结果
inputs = {
    "input": "2025年1月15日杭州什么天气?",
    "chat_history": [],
    "agent_outcome": None,
    "intermediate_steps": []
}
print("\n=== 开始执行图 ===")
print("初始输入:", inputs)
result = app.invoke(inputs)
print("\n=== 执行结束 ===")
if isinstance(result["agent_outcome"], AgentFinish):
    print("最终 Agent 输出消息:", result["agent_outcome"].return_values["output"])
else:
    print("最终 Agent 输出消息:", result["agent_outcome"])
