from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

class FeedbackState(TypedDict, total=False):
    user_input: str
    sentiment: str # "positive", "negative", "neutral"
    response: str
    final_out:str
    confidence: float    # 情绪判断置信度（Pydantic模式使用）
    reason: str          # 情绪判断原因（Pydantic模式使用）
# ==== 节点 ====

def analyze_sentiment(state: FeedbackState) -> dict:
    """情绪分析（简化版，实际上可用LLM）"""
    text = state["user_input"].lower()

    positive_words = ["好","棒","满意","喜欢","excellent","good", "great", "happy", "love", "like"]
    negative_words = ["差","不满","烂","投诉","bad", "terrible", "hate", "angry", "sad"]

    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)

    if pos_count > neg_count:
        sentiment = "positive"
    elif neg_count > pos_count:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {"sentiment": sentiment}

# ==== 配合LLM做情绪分析====
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="deepseek-v4-flash", temperature=0)

def analyze_sentiment_with_llm(state: FeedbackState) -> dict:
    prompt = f"""请分析以下用户反馈的文本情绪，只能回复 'positive', 'negative', 'neutral' 中的一个：
    
文本：{state['user_input']}
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    sentiment = response.content.strip().lower()

    if sentiment not in ["positive", "negative", "neutral"]:
        sentiment = "neutral"
    return {"sentiment": sentiment}


# Pydantic 结构化输出
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
import json

class SentimentOutput(BaseModel):
    sentiment: str # "positive", "negative", "neutral"
    confidence: float # 0.0 ~ 1.0
    reason: str # 判断原因

llm = ChatOpenAI(model="deepseek-v4-flash")
# structured_llm = llm.with_structured_output(SentimentOutput)

def analyze_sentiment_pydantic(state: FeedbackState) -> dict:
    # reasult = structured_llm.invoke([HumanMessage(content=state['user_input'])])
    # return {"sentiment": reasult.sentiment, "confidence": reasult.confidence, "reason": reasult.reason}

    prompt = f"""请分析以下用户反馈的情绪，并严格按照 JSON 格式返回：
    文本：{state['user_input']}
    返回格式（严格遵守，不要添加其他内容）：
    {{"sentiment": "positive/negative/neutral", "confidence": 0.0~1.0, "reason": "判断原因"}}
    """
    response = llm.invoke(prompt)
    # 解析 JSON 字符串并转换为 Pydantic 对象
    result_dict = json.loads(response.content.strip())
    result = SentimentOutput(**result_dict)
    return {"sentiment": result.sentiment, "confidence": result.confidence, "reason": result.reason}

def handle_positive(state: FeedbackState) -> dict:
    """处理正面反馈"""
    response = f"感谢您的好评！很高兴我们的服务让您满意。您的反馈：{state['user_input']}"
    return {"response": response}

def handle_negative(state: FeedbackState) -> dict:
    """处理负面反馈"""
    response = (
        f"非常抱歉给您带来了不好的体验！"
        f"您的反馈已记录：'{state['user_input']}'。"
        f"我们的人工客服将在24小时内联系您。"
    )
    return {"response": response}

def handle_neutral(state: FeedbackState) -> dict:
    """处理中性反馈"""
    response = f"感谢您的反馈：'{state['user_input']}'。我们会继续改进服务。"
    return {"response": response}

def format_output(state: FeedbackState) -> dict:
    """格式化最终输出"""
    final_out = f"情绪分析结果: {state['sentiment']}\n客服回复: {state['response']}"
    return {"final_out": final_out}

# ==== 路由函数 ====

def route_by_sentiment(state: FeedbackState)->str:
    """根据情绪返回路由 key"""
    return state["sentiment"]

# # ==== 构建状态图 ====

# graph = StateGraph(FeedbackState)

# graph.add_node("analyze_sentiment", analyze_sentiment)
# graph.add_node("handle_positive", handle_positive)
# graph.add_node("handle_negative", handle_negative)
# graph.add_node("handle_neutral", handle_neutral)
# graph.add_node("format_output", format_output)

# graph.set_entry_point("analyze_sentiment")

# # 添加条件边
# graph.add_conditional_edges(
#     "analyze_sentiment",
#     route_by_sentiment,
#     {
#         "positive": "handle_positive",
#         "negative": "handle_negative",
#         "neutral": "handle_neutral"
#     }
# )

# # 三个分支最终都汇入 format
# graph.add_edge("handle_positive", "format_output")
# graph.add_edge("handle_negative", "format_output")
# graph.add_edge("handle_neutral", "format_output")

# graph.add_edge("format_output", END)


# app = graph.compile()

# # 运行
#  #正面反馈
# r=app.invoke({
#     "user_input": "产品很棒，服务也很满意！",
#     "sentiment": "",
#     "response": "",
#     "final_out": ""
# })

# print(r["final_out"])

#  #负面反馈
# r=app.invoke({
#     "user_input": "太差了，完全不满意，要投诉！",
#     "sentiment": "",
#     "response": "",
#     "final_out": ""
# })

# print(r["final_out"])

# # ==== 构建状态图_LLM ====

# graph = StateGraph(FeedbackState)

# graph.add_node("analyze_sentiment", analyze_sentiment_with_llm)
# graph.add_node("handle_positive", handle_positive)
# graph.add_node("handle_negative", handle_negative)
# graph.add_node("handle_neutral", handle_neutral)
# graph.add_node("format_output", format_output)

# graph.set_entry_point("analyze_sentiment")

# # 添加条件边
# graph.add_conditional_edges(
#     "analyze_sentiment",
#     route_by_sentiment,
#     {
#         "positive": "handle_positive",
#         "negative": "handle_negative",
#         "neutral": "handle_neutral"
#     }
# )

# # 三个分支最终都汇入 format
# graph.add_edge("handle_positive", "format_output")
# graph.add_edge("handle_negative", "format_output")
# graph.add_edge("handle_neutral", "format_output")

# graph.add_edge("format_output", END)


# app = graph.compile()

# # 运行
#  #正面反馈
# r=app.invoke({
#     "user_input": "产品很棒，服务也很满意！",
#     "sentiment": "",
#     "response": "",
#     "final_out": ""
# })

# print(r["final_out"])

#  #负面反馈
# r=app.invoke({
#     "user_input": "太差了，完全不满意，要投诉！",
#     "sentiment": "",
#     "response": "",
#     "final_out": ""
# })

# print(r["final_out"])


# ==== 构建状态图_Pydantic ====

graph = StateGraph(FeedbackState)

graph.add_node("analyze_sentiment", analyze_sentiment_pydantic)
graph.add_node("handle_positive", handle_positive)
graph.add_node("handle_negative", handle_negative)
graph.add_node("handle_neutral", handle_neutral)
graph.add_node("format_output", format_output)

graph.set_entry_point("analyze_sentiment")

# 添加条件边
graph.add_conditional_edges(
    "analyze_sentiment",
    route_by_sentiment,
    {
        "positive": "handle_positive",
        "negative": "handle_negative",
        "neutral": "handle_neutral"
    }
)

# 三个分支最终都汇入 format
graph.add_edge("handle_positive", "format_output")
graph.add_edge("handle_negative", "format_output")
graph.add_edge("handle_neutral", "format_output")

graph.add_edge("format_output", END)


app = graph.compile()

# 运行
 #正面反馈
r=app.invoke({
    "user_input": "产品很棒，服务也很满意！",
    "sentiment": "",
    "response": "",
    "final_out": ""
})

print(r["final_out"])

 #负面反馈
r=app.invoke({
    "user_input": "太差了，完全不满意，要投诉！",
    "sentiment": "",
    "response": "",
    "final_out": ""
})

print(r["final_out"])


for step in app.stream({
    "user_input": "产品很棒，服务也很满意！",
    "sentiment": "",
    "response": "",
    "final_out": ""
}):
    node_name, state_update = list(step.items())[0]
    print(f"节点: {node_name}, 输出: {state_update}")


for step in app.stream({
    "user_input": "太差了，完全不满意，要投诉！",
    "sentiment": "",
    "response": "",
    "final_out": ""
}):
    node_name, state_update = list(step.items())[0]
    print(f"节点: {node_name}, 输出: {state_update}")
