from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1.定义State
class TextState(TypedDict):
    raw_text: str
    cleaned: str
    word_count: int
    summary: str

# 2.定义节点
def clean_node(state:TextState)->dict:
    """清理文本，去掉多余空格"""
    text = state["raw_text"].strip()
    return {"cleaned": text}

def count_node(state:TextState)->dict:
    """统计词数"""
    count = len(state["cleaned"].split())
    return {"word_count": count}

def summary_node(state:TextState)->dict:
    """生成摘要"""
    text = state["cleaned"]
    summary = text[:50] + "..." if len(text) > 50 else text
    return {"summary": summary}

# 3.组装图
graph = StateGraph(TextState)
graph.add_node("clean", clean_node)
graph.add_node("count", count_node)
graph.add_node("summarize", summary_node)

graph.set_entry_point("clean")
graph.add_edge("clean","count")
graph.add_edge("count","summarize")
graph.add_edge("summarize",END)

app = graph.compile()

# 4.运行
result = app.invoke({
    "raw_text": "  LangGraph 是一个用于构建有状态 Agent 应用的框架，基于图结构描述执行流。  ",
    "cleaned": "",
    "word_count": 0,
    "summary": "",
})

print("清理后的文本：", result["cleaned"])
print("词数：", result["word_count"])
print("摘要：", result["summary"])

# ASCII 可视化
print(app.get_graph().draw_ascii())