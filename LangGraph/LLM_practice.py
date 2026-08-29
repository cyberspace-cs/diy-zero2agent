import os 
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import json
import dotenv

import time
from langchain_core.exceptions import OutputParserException

dotenv.load_dotenv()

#初始化（在节点外初始化，避免每次调用的都重建）
llm = ChatOpenAI(
    model="deepseek-v4-flash",
    temperature=0.7,
    api_key=os.environ["OPENAI_API_KEY"],
)

# ==== 定义状态 ====
class ContentState(TypedDict, total=False):
    topic: str
    keywords: str
    outline: str
    article: str
    prompt: str
# ==== 定义节点 ====
def extract_keywords(state: ContentState)->dict:
    """提取关键词"""
    response = llm.invoke([
        HumanMessage(content=f"从主题'{state['topic']}'中提取5个核心关键词，用逗号分隔，只输出关键词：")
    ])
    return {"keywords": response.content.strip()}

def create_outline(state: ContentState)->dict:
    """根据关键词生成大纲"""
    response = llm.invoke([
        HumanMessage(content=((
            f"基于主题'{state['topic']}'和关键词'{state['keywords']}'"
            f"生成一个4节的文章大纲，每节一行，格式为：数字.标题"
        )))
    ])
    return {"outline": response.content.strip()}

def write_article(state: ContentState)->dict:
    """生成文章内容"""
    response = llm.invoke([
        HumanMessage(content=(
            f"基于主题'{state['topic']}'和关键词'{state['keywords']}'，"
            f"根据大纲'{state['outline']}'写出完整的文章"
        )
    )])
    
    return {"article": response.content.strip()}

def robust_llm_node(state: dict)->dict:
    """带重试的LLM节点"""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = llm.invoke([HumanMessage(content=state["prompt"])])
            return {"result": response.content}
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)#指数退避，每次失败后等待2^attempt秒
                continue
            else:
                return {"result": f"错误: {str(e)}","error":True}

#===== 构建状态图 ====

graph = StateGraph(ContentState)

graph.add_node("extract_keywords", extract_keywords)
graph.add_node("create_outline", create_outline)
graph.add_node("write_article", write_article,)

graph.set_entry_point("extract_keywords")
graph.add_edge("extract_keywords", "create_outline")
graph.add_edge("create_outline", "write_article")
graph.add_edge("write_article", END)

app=graph.compile()

if __name__=="__main__":
    result=app.invoke(
        {
            "topic":"LangGraph状态图编程",
            "keywords":"",
            "outline":"",
            "article":"",
        }
    )
    print("关键词:",result["keywords"])
    print("\n大纲内容:\n",result["outline"])
    print("\n文章内容:\n",result["article"])
    with open("LangGraph_programing.md","w" ,encoding="utf-8") as f:
        f.write(result["article"])


# 示例
# def summarize_node(state: dict)->dict:
#     """在节点里面调用ChatOpenAI"""
#     text = state["input_text"]

#     response = llm.invoke([
#         SystemMessage(content="你现在是一个专业的文本摘要助手"),
#         HumanMessage(content=f"请根据下面的内容进行总结:\n\n{text}")
#     ])

#     return {"summary":response.content}

# from langchain_core.prompts import ChatPromptTemplate

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "你现在是一个专业的{role}"),
#     ("human", "{task}"),
# ])

# chain = prompt_template | llm

# def analyze_node(state: dict)->dict:
#     response = chain.invoke({
#         "role": "代码审查专家",
#         "task": f"审查以下代码：\n{state['code']}",
#     })
#     return {"review":response.content}

# from pydantic import BaseModel, Field
# from typing import List

# class CodeReview(BaseModel):
#     score: int = Field(description="代码质量评分1-10")
#     issues: List[str] = Field(description="发现的问题列表")
#     suggestions: List[str] = Field(description="改进建议")


# def review_node(state: dict)->dict:
#     response = llm.invoke(
#         f"请审查这段代码并给出评分：\n{state['code']}"
#     )
#     result_dict = json.loads(response.content.strip())
#     result: CodeReview = CodeReview(**result_dict)

#     return {
#         "score": result.score,
#         "issues": result.issues,
#         "suggestions": result.suggestions,
#     }
