from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import StateGraph, END

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()

from huggingface_hub import InferenceClient




# ==== State ====
class BlogState(TypedDict):
    topic: str
    outline: str
    draft: str
    final_article: str

# ==== 节点（不依赖特定LLM，方便替换） ====

def generate_outline(state: BlogState) -> dict:
    """第一步：生成文章大纲"""
    topic = state["topic"]
    prompt = f"""为以下主题创建一个清晰的博客文章大纲：
    主题：{topic}

    请提供：
    1.引言要点
    2.3-4个主要章节标题
    3.结论要点

    大纲："""
    # outline = llm.invoke(prompt) #替换为真实LLM调用
    # 示例输出（演示用）
    outline = f"""
## {topic} 完整指南

**引言**：介绍{topic}的背景和重要性

**第一章：基础概念**
- 核心定义
- 关键术语

**第二章：实现方式**
- 主流方案对比
- 最佳实践

**第三章：实战案例**
- 具体示例
- 常见问题

**结论**：总结要点，给出建议
""".strip()

    return {"outline": outline}

def generate_outline_openai(state: BlogState) -> dict:
    topic = state["topic"]
    prompt = f"为主题'{topic}创建博客大纲（3-4个章节）"
    llm = ChatOpenAI(model="deepseek-v4-flash", temperature=0.7)

    outline = llm.invoke([HumanMessage(content=prompt)]).content
    return {"outline": outline}



def generate_outline_huggingface(state: BlogState) -> dict:
    topic = state["topic"]
    prompt = f"为主题'{topic}'创建博客大纲（3-4个章节）:"

    #需要HuggingFace API token
    client=InferenceClient(
        api_key=os.environ["HFUGGINGFACEHUB_API_TOKEN"],
    )

    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Flash-0731:novita",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    response = completion.choices[0].message
    outline = response.content.strip()
    return {"outline": outline}

def expand_content(state: BlogState) -> dict:
    """第二步：根据大纲生成正文草稿"""
    outline = state["outline"]
    topic = state["topic"]

    prompt = f"""根据以下大纲，为主题"{topic}"写一篇详细的博客文章草稿。
每个章节至少写 2-3 段，包含具体示例。


    大纲：
    {outline}

    文章草稿："""
    # draft = llm.invoke(prompt) #替换为真实LLM调用
    
    draft = f"""# {topic}完整指南

### 引言

{topic}是现代软件开发中的重要话题...（正文草稿）

## 基础概念

理解{topic}首先需要掌握几个核心概念...

## 实现方式

在实际项目中，有多种方式可以实现{topic}...

## 实战案例

以下是一个典型的{topic}应用场景


### 结论

通过本文的介绍，我们深入了解了{topic}的各个方面...
""".strip()

    return {"draft": draft}


def expand_content_openai(state: BlogState) -> dict:
    """第二步：根据大纲生成正文草稿"""
    outline = state["outline"]
    topic = state["topic"]

    prompt = f"""根据以下大纲，为主题"{topic}"写一篇详细的博客文章草稿。
每个章节至少写 2-3 段，包含具体示例。


    大纲：
    {outline}

    文章草稿："""
    llm = ChatOpenAI(model="deepseek-v4-flash", temperature=0.7)

    draft = llm.invoke([HumanMessage(content=prompt)]).content.strip() #替换为真实LLM调用
    return {"draft": draft}

def expand_content_huggingface(state: BlogState) -> dict:
    """第二步：根据大纲生成正文草稿"""
    outline = state["outline"]
    topic = state["topic"]

    prompt = f"根据以下大纲，为主题'{topic}'写一篇详细的博客文章草稿。每个章节至少写 2-3 段，包含具体示例。\n\n 大纲:\n{outline}"
    #需要HuggingFace API token
    client=InferenceClient(
        api_key=os.environ["HFUGGINGFACEHUB_API_TOKEN"],
    )
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Flash-0731:novita",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    response = completion.choices[0].message
    draft = response.content.strip() #替换为真实LLM调用
    return {"draft": draft}

def polish_article(state: BlogState) -> dict:
    """润色文章草稿，使其更加流畅和专业。

    对生成的文章草稿进行润色和优化，提升语言流畅度和逻辑清晰度，
    使其更适合技术博客读者的阅读习惯。

    Args:
        state: 博客状态字典，包含"draft"键，值为待润色的文章草稿

    Returns:
        包含润色后最终文章的字典，键为"final_article"
    """
    draft = state["draft"]

    prompt = f"""请对以下文章草稿进行润色和优化，使其：
1. 语言更流畅自然
2. 逻辑更清晰
3. 适合技术博客读者

文章草稿：
{draft}

润色后的文章："""
    # final = llm.invoke(prompt) #替换为真实LLM调用
    final = draft+"\n\n---\n*（已润色优化）*"

    return {"final_article": final}

#润色文章草稿，使其更加流畅和专业。 OPENAI
def polish_article_openai(state: BlogState) -> dict:
    """润色文章草稿，使其更加流畅和专业。"""
    draft = state["draft"]
    
    
    prompt = f"""请对以下文章草稿进行润色和优化，使其：
1. 语言更流畅自然
2. 逻辑更清晰
3. 适合技术博客读者

文章草稿：
{draft}

润色后的文章："""
    
    llm = ChatOpenAI(model="deepseek-v4-flash", temperature=0.7)
    final = llm.invoke([HumanMessage(content=prompt)]).content.strip() #替换为真实LLM调用

    return {"final_article": final}

def polish_article_huggingface(state: BlogState) -> dict:
    """润色文章草稿，使其更加流畅和专业。"""
    draft = state["draft"]
    prompt = f"[INST] 请对以下文章草稿进行润色和优化，使其：1. 语言更流畅自然 2. 逻辑更清晰 3. 适合技术博客读者。[/INST]\n\n 文章草稿:\n{draft}\n\n 润色后的文章:"
    #需要HuggingFace API token
    client=InferenceClient(
        api_key=os.environ["HFUGGINGFACEHUB_API_TOKEN"],
    )
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Flash-0731:novita",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )
    response = completion.choices[0].message
    final = response.content.strip() #替换为真实LLM调用

    return {"final_article": final}

# 检查大纲质量，路由函数
def check_outline_quality(state: BlogState) -> str:
    """检查大纲质量，决定是直接展开还是重新生成大纲。"""
    outline = state["outline"]

    # 简单检查：大纲是否包含足够的章节
    if outline.count('##')>=3:
        return "expand" #质量够，继续展开
    else:
        return "regenerate_outline" #质量不够，重新生成大纲



# ==== 构建状态图 ====
graph = StateGraph(BlogState)

graph.add_node("generate_outline", generate_outline_openai)
graph.add_node("expand_content", expand_content_openai)
graph.add_node("polish_article", polish_article_openai)


graph.set_entry_point("generate_outline")

graph.add_conditional_edges("generate_outline", 
                            check_outline_quality,
                            {
                                "expand": "expand_content",
                                "regenerate_outline": "generate_outline",#循环回去重新生成
                            }
)

graph.add_edge("expand_content", "polish_article")
graph.add_edge("polish_article", END)

app=graph.compile()

result = app.invoke({
    "topic":"LangGraph入门指南",
    "outline":"",
    "draft":"",
    "final_article":"",
})

print("=== 大纲 ===")
print(result["outline"])
print("\n=== 最终文章 ===")
with open("final_article_openai.md", "w", encoding="utf-8") as f:
    f.write("# LangGraph 入门指南(OPENAI)\n\n")
    f.write(result["final_article"])
print(result["final_article"])

                 
# for step in app.stream({
#     "topic":"LangGraph入门指南",
#     "outline":"",
#     "draft":"",
#     "final_article":"",
# }):
#     node_name = list(step.keys())[0]
#     print(f"\n--- [{node_name}完成 ---]")

#     if node_name == "generate_outline":
#         print(step["generate_outline"]["outline"][:200])
#     elif node_name == "expand_content":
#         print(step["expand_content"]["draft"][:200])
#     elif node_name == "polish_article":
#         print(step["polish_article"]["final_article"][:200])
#         print("文章已经润色完成")



# ==== 构建状态图 ====
graph = StateGraph(BlogState)

graph.add_node("generate_outline", generate_outline_huggingface)
graph.add_node("expand_content", expand_content_huggingface)
graph.add_node("polish_article", polish_article_huggingface)


graph.set_entry_point("generate_outline")

graph.add_conditional_edges("generate_outline", 
                            check_outline_quality,
                            {
                                "expand": "expand_content",
                                "regenerate_outline": "generate_outline",#循环回去重新生成
                            }
)

graph.add_edge("expand_content", "polish_article")
graph.add_edge("polish_article", END)

app=graph.compile()

result = app.invoke({
    "topic":"LangGraph入门指南",
    "outline":"",
    "draft":"",
    "final_article":"",
})

print("=== 大纲 ===")
print(result["outline"])
print("\n=== 最终文章 ===")

with open("final_article_huggingface.md", "w", encoding="utf-8") as f:
    f.write("# LangGraph 入门指南(HUGGINGFACE)\n\n")
    f.write(result["final_article"])
print(result["final_article"])
                 
# for step in app.stream({
#     "topic":"LangGraph入门指南",
#     "outline":"",
#     "draft":"",
#     "final_article":"",
# }):
#     node_name = list(step.keys())[0]
#     print(f"\n--- [{node_name}完成 ---]")

#     if node_name == "generate_outline":
#         print(step["generate_outline"]["outline"][:200])
#     elif node_name == "expand_content":
#         print(step["expand_content"]["draft"][:200])
#     elif node_name == "polish_article":
#         print(step["polish_article"]["final_article"][:200])
#         print("文章已经润色完成")
