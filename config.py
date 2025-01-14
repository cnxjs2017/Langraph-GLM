from langchain_core.prompts import MessagesPlaceholder
from langgraph.graph import END


class Config:
    # 这里要用你自己的 API 密钥
    API_KEY = ""

    # 模型配置
    MODEL_CONFIG = {
        "temperature": 0.8,
        "model": "glm-4-flash",
    }

    # 工具配置
    TOOLS = ["web_search"]

    # Prompt 模板配置
    PROMPT_TEMPLATES = {
        "agent_prompt": [
            ("system", "你是一个有用的助手。当你无法回答问题时，请调用工具来获取信息。"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ],
        "beautify_prompt": [
            ("system", "你是一个文笔优化助手，负责对文本进行润色和美化，使其更加流畅和优雅。"),
            ("user", "请对以下文本进行文笔优化：\n{text}"),
        ],
    }

    # Workflow 顺序配置
    WORKFLOW_ORDER = [
        "agent",  # 第一步：运行主 Agent
        "action",  # 第二步：执行工具
        "beautify",  # 第三步：美化输出
    ]

    # 条件判断配置
    CONDITIONAL_EDGES = {
        "agent": {
            "continue": "action",  # 如果 Agent 需要调用工具，跳转到 action
            "beautify": "beautify",  # 如果 Agent 完成，跳转到 beautify
            "end": END,  # 如果未知情况，结束流程
        },
        "action": "agent",  # 执行工具后，返回 agent
        "beautify": END,  # 美化输出后，结束流程
    }
