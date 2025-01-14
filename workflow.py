import uuid
from typing import TypedDict, Union, List, Annotated
import operator
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langchain.agents.output_parsers.tools import ToolAgentAction
from agents import agent_runnable, beautify_agent
from tools import tools
from config import Config


# 定义状态字典
class AgentState(TypedDict):
    input: str
    chat_history: list[BaseMessage]
    agent_outcome: Union[AgentAction, AgentFinish, None]
    intermediate_steps: Annotated[List[tuple[AgentAction, str]], operator.add]


# 创建工具节点
tool_node = ToolNode(tools)


# 定义节点函数
def run_agent(data: AgentState) -> dict:
    print("\n--- 进入 agent 节点 ---")
    print("当前状态:", data)
    agent_outcome = agent_runnable.invoke(data)
    if isinstance(agent_outcome, list):
        agent_outcome = agent_outcome[0]
    print("Agent 输出结果:", agent_outcome)
    return {"agent_outcome": agent_outcome}


def execute_tools(data: AgentState) -> dict:
    print("\n--- 进入 action 节点 ---")
    print("当前状态:", data)
    agent_action = data["agent_outcome"]
    if isinstance(agent_action, list):
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
                {"role": "user", "content": tool_input},
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": tool_call_id,
                            "name": tool_name,
                            "args": {"query": tool_input},
                        }
                    ],
                ),
            ],
            "tool": tool_name,
            "tool_input": tool_input,
        }

        output = tool_node.invoke(tool_input_dict)
        print("工具执行结果:", output)
    else:
        output = "Unknown action"
    return {"intermediate_steps": data["intermediate_steps"] + [(agent_action, str(output))]}


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
    if isinstance(action, AgentFinish):
        print("Agent 完成，进入美化节点")
        return "beautify"
    elif isinstance(action, ToolAgentAction):
        print(f"Agent 需要调用工具: {action.tool}, 输入: {action.tool_input}")
        return "continue"
    else:
        print("未知的 Agent 输出，结束执行")
        return "end"


# 创建和配置图
def create_workflow():
    workflow = StateGraph(AgentState)

    # 添加节点
    for node_name in Config.WORKFLOW_ORDER:
        if node_name == "agent":
            workflow.add_node("agent", run_agent)
        elif node_name == "action":
            workflow.add_node("action", execute_tools)
        elif node_name == "beautify":
            workflow.add_node("beautify", beautify_output)

    # 设置入口点
    workflow.set_entry_point(Config.WORKFLOW_ORDER[0])

    # 添加条件边
    for node_name, edges in Config.CONDITIONAL_EDGES.items():
        if isinstance(edges, dict):
            workflow.add_conditional_edges(node_name, should_continue, edges)
        else:
            workflow.add_edge(node_name, edges)

    return workflow.compile()
