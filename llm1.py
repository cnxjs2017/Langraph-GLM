from typing import Annotated
from typing_extensions import TypedDict

from langchain_community.chat_models import ChatZhipuAI
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

chat_zhipu = ChatZhipuAI(
    api_key="",  # 这里填自己的api_key
    model="glm-4-flash",
    temperature=0.8,
)


def chatbot(state: State):
    response = chat_zhipu.invoke(state["messages"])
    print("response:", response)
    return {"messages": [response]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph = graph_builder.compile()

initial_state = {
    "messages": [
        {"role": "user", "content": "嗨！"}
    ]
}
result = graph.invoke(initial_state)
print(result["messages"][-1].content)
