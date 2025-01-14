import requests
import uuid
from langchain.tools import tool
from config import Config


# 定义联网搜索工具
@tool("web_search")
def web_search(query: str) -> str:
    """
    调用联网搜索工具，返回搜索结果。
    """
    print("query:", query)
    tool_name = "web-search-pro"
    url = "https://open.bigmodel.cn/api/paas/v4/tools"
    request_id = str(uuid.uuid4())
    data = {
        "request_id": request_id,
        "tool": tool_name,
        "stream": False,
        "messages": [{"role": "user", "content": query}]
    }
    headers = {'Authorization': Config.API_KEY}
    resp = requests.post(url, json=data, headers=headers, timeout=300)
    print("联网搜索工具调用结果:", resp.status_code)
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
