from langchain_community.chat_models import ChatZhipuAI
from langchain.agents import create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from config import Config
from tools import tools

# 初始化 ChatZhipuAI 模型
chat_zhipu = ChatZhipuAI(
    temperature=Config.MODEL_CONFIG["temperature"],
    api_key=Config.API_KEY,
    model=Config.MODEL_CONFIG["model"],
)

# 从配置中加载 Prompt 模板
agent_prompt = ChatPromptTemplate.from_messages(Config.PROMPT_TEMPLATES["agent_prompt"])
beautify_prompt = ChatPromptTemplate.from_messages(Config.PROMPT_TEMPLATES["beautify_prompt"])

# 创建 Agent
agent_runnable = create_openai_tools_agent(chat_zhipu, tools, agent_prompt)

# 创建美化 Agent
beautify_agent = beautify_prompt | chat_zhipu