from workflow import create_workflow

# 初始化工作流
app = create_workflow()

# 运行图并获取结果
inputs = {
    "input": "帮我找一找近期的离婚相关案件，并给出对应的链接",
    "chat_history": [],
    "agent_outcome": None,
    "intermediate_steps": []
}
print("\n=== 开始执行图 ===")
print("初始输入:", inputs)
result = app.invoke(inputs)
print("执行结果:", result)
print("\n=== 执行结束 ===")
