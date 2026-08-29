# LangGraph 入门指南(HUGGINGFACE)

# LangGraph 入门指南：从零构建真正可控的 AI Agent

近年来，以 GPT 为代表的大语言模型（LLM）不断拓展 AI 应用的能力边界。然而，当我们真正着手构建面向实际场景的智能体（Agent）时，一个核心问题便会浮现：**如何管理多步执行中的状态？如何让 Agent 在循环、分支、暂停与恢复之间保持可控？**

LangGraph 正是为解决这些问题而生的编排框架。它由 LangChain 团队开发，将图结构引入 AI 工作流设计，使开发者能够像绘制流程图一样定义 Agent 的行为。本文将从零开始，带你理解 LangGraph 的核心思想，并通过大量代码示例掌握其基本用法与进阶技巧。

---

## 第一章：LangGraph 是什么？——核心概念与架构

### 为什么需要 LangGraph

设想一个典型场景：你希望 Agent 完成“查询资料 → 总结 → 生成报告 → 发给用户”这一流程。使用普通代码，你可能会这样写：

```python
data = search(keyword)
summary = llm.summarize(data)
report = llm.generate_report(summary)
send(report)
```

这段代码看起来清晰，但真实的 Agent 往往需要根据中间结果动态调整下一步。例如，当总结不够完整时，Agent 需要重新搜索；当用户对结果不满意时，Agent 需要重新生成。此外，你可能还希望在任意步骤暂停，等待人工审核，然后继续执行。传统的线性代码很难优雅地支持这些能力。

LangGraph 的核心设计逻辑是：**将工作流建模为一个有向图（Graph），节点（Node）负责计算，边（Edge）决定流转路径，全局状态（State）在节点之间传递和更新。** 这种架构天然支持循环、条件分支、持久化和人机协同，让复杂的 AI 应用变得可维护、可调试、可复现。

### LangGraph 与 LangChain、Agent 的关系

很多人将 LangGraph 与 LangChain 混为一谈，其实它们的分工并不相同。

- **LangChain** 是一个 LLM 应用开发工具包，提供模型封装、Prompt 管理、文档加载、输出解析等基础能力。
- **LangGraph** 是建立在 LangChain 之上的编排层，专注于构建和控制多步骤、有状态的 Agent 工作流。
- **Agent** 则是一种应用模式：通过“思考-行动-观察”的循环来完成任务。LangGraph 是少数能精细控制这种循环的框架之一。

可以这样理解：LangChain 提供零件，LangGraph 负责将这些零件组装成一台有逻辑、有状态的机器。你完全可以不使用 LangChain，直接用 LangGraph 调用任何 Python 函数或外部 API，但两者结合使用体验更佳。

### 核心概念图解

LangGraph 的五个核心概念可以这样理解：

- **State（全局状态）**：一个在图内流动的数据结构，通常是 `TypedDict` 或 Pydantic 模型。每个节点都可以读取和修改 State。
- **Node（节点）**：一个 Python 函数，接收当前 State，返回更新后的 State 片段。
- **Edge（边）**：节点之间的连接，决定执行顺序。LangGraph 还支持条件边（conditional edges），可根据 State 动态选择下一步节点。
- **Graph（图）**：由节点和边组成的整体编排对象，必须通过 `compile()` 编译后才能运行。
- **Checkpointer（检查点）**：可选组件，用于保存图执行的每一步状态，从而实现持久化和断点恢复。

下面是一个最小的概念映射代码：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    count: int

def add_one(state: State):
    return {"count": state["count"] + 1}

builder = StateGraph(State)
builder.add_node("add_one", add_one)
builder.add_edge(START, "add_one")
builder.add_edge("add_one", END)
graph = builder.compile()
```

### 关键特性

LangGraph 之所以能迅速走红，是因为它解决了很多实际痛点：

- **循环（Loops）**：支持从一个节点跳回之前的节点，这是 Agent 实现反思、迭代的基础。
- **条件分支（Conditional Branches）**：根据 State 中的值或 LLM 的输出来决定走哪条边。
- **持久化（Persistence）**：每次执行均可保存为检查点，重启后还能恢复到任意一步。
- **人机协同（Human-in-the-Loop）**：可以在节点之间插入人工审核步骤，让系统先暂停等待人工反馈，再继续执行。

这些特性组合起来，使 LangGraph 成为构建生产级 Agent 的有力工具。

---

## 第二章：环境搭建与第一个 Graph

### 安装 LangGraph

开始之前，请确保已安装 Python 3.9 或更高版本。LangGraph 的安装非常简单：

```bash
pip install langgraph
```

如果你还需要调用 OpenAI、Anthropic 或其他模型，可以安装对应的 LangChain 集成包：

```bash
pip install langchain-openai langchain-anthropic
```

如需将图结构导出为图片，建议再安装：

```bash
pip install "langgraph[draw]"  # 或者安装 pygraphviz，用于生成 PNG
```

另外，请准备好模型 API Key，并通过环境变量进行设置：

```bash
export OPENAI_API_KEY="sk-..."
# 或
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 最小示例：输入 → 处理 → 输出

我们从最简单的流程开始：接收用户输入，做字符串处理，然后输出结果。该示例虽简单，却完整体现了 LangGraph 的“三步走”：**定义 State → 实现节点函数 → 构建并编译 Graph**。

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# 1. 定义 State
class State(TypedDict):
    user_input: str
    processed: str

# 2. 定义节点函数
def process_node(state: State):
    # 模拟业务逻辑：加上前缀
    return {"processed": f"LangGraph 已收到：{state['user_input']}"}

def output_node(state: State):
    print(f"最终输出: {state['processed']}")
    return {}

# 3. 构建并编译图
builder = StateGraph(State)
builder.add_node("process", process_node)
builder.add_node("output", output_node)

builder.add_edge(START, "process")
builder.add_edge("process", "output")
builder.add_edge("output", END)

graph = builder.compile()

# 4. 运行
result = graph.invoke({"user_input": "你好，世界！"})
print(result)
```

运行这段代码，你会看到：

```
最终输出: LangGraph 已收到：你好，世界！
{'user_input': '你好，世界！', 'processed': 'LangGraph 已收到：你好，世界！'}
```

### 代码拆解

**定义 State**：这里的 State 是一个 `TypedDict`，包含两个字段：`user_input` 用于接收初始数据，`processed` 用于保存中间结果。LangGraph 允许节点只返回 State 的子集，它会将返回值与当前 State 合并。因此，你不需要在每个节点里返回所有字段。

**实现节点函数**：每个节点就是一个普通 Python 函数，参数是当前 State，返回值是一个字典，表示要更新的字段。上述 `process_node` 只返回了 `processed`，并未修改 `user_input`，但 LangGraph 会自动保留原始输入。

**构建并编译 Graph**：通过 `StateGraph(State)` 创建图构建器，依次添加节点和边。`START` 和 `END` 是特殊节点，分别表示入口和出口。最后调用 `compile()` 得到可执行的 `CompiledGraph` 对象。

`invoke()` 是同步运行 Graph 的入口。你还可以使用 `ainvoke()` 支持异步运行，或使用 `stream()` 逐节点输出执行过程中的中间状态。

### 运行与调试：可视化工具

对于复杂图，仅靠脑补很难跟踪执行流程。LangGraph 提供了可视化能力，可以直接生成 Mermaid 格式的图：

```python
# 在 Jupyter Notebook 中可以直接渲染
graph.get_graph().draw_mermaid()

# 也可以输出 Mermaid 字符串，复制到 mermaid.live 渲染
print(graph.get_graph().draw_mermaid())
```

如果安装了 `pygraphviz`，还可以直接生成图片：

```python
graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
```

运行完图后，可以使用 `graph.get_state(config)` 查看某个检查点的状态，这在调试时非常有用。下一章将详细介绍持久化配置。

---

## 第三章：核心进阶——循环、条件与记忆

### 用 `add_conditional_edges` 实现动态路由

实际应用中，我们经常需要根据当前状态决定下一步走哪个分支。LangGraph 的条件边允许用一个路由函数动态选择后续节点。

例如，实现一个“情感分类 → 不同处理”的流程：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class ReviewState(TypedDict):
    review: str
    sentiment: str

def analyze_sentiment(state: ReviewState):
    # 简单模拟：如果包含“好”“棒”则判定为 positive
    if "好" in state["review"]:
        return {"sentiment": "positive"}
    return {"sentiment": "negative"}

def handle_positive(state: ReviewState):
    return {"sentiment": f"好评处理: {state['review']}"}

def handle_negative(state: ReviewState):
    return {"sentiment": f"差评处理: {state['review']}"}

# 路由函数：返回下一步节点名
def route_by_sentiment(state: ReviewState) -> str:
    if state["sentiment"] == "positive":
        return "handle_positive"
    return "handle_negative"

builder = StateGraph(ReviewState)
builder.add_node("analyze_sentiment", analyze_sentiment)
builder.add_node("handle_positive", handle_positive)
builder.add_node("handle_negative", handle_negative)

builder.add_edge(START, "analyze_sentiment")
builder.add_conditional_edges("analyze_sentiment", route_by_sentiment)
builder.add_edge("handle_positive", END)
builder.add_edge("handle_negative", END)

graph = builder.compile()
graph.invoke({"review": "这个产品真好"})
```

`add_conditional_edges` 的第一个参数是源节点，第二个参数是路由函数。路由函数必须返回目标节点名称。你也可以传入一个映射字典，将返回值映射到具体节点名，这样更安全。

### 循环结构与 Agent 的“思考-行动-观察”闭环

Agent 的本质是一个循环：模型决定调用哪些工具 → 执行工具 → 把结果反馈给模型 → 模型继续决定下一步。LangGraph 对循环的支持非常自然。

以下是一个简化的 ReAct（Reasoning + Acting）Agent 核心结构：

```python
from typing import TypedDict, Annotated
from operator import add
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

class AgentState(TypedDict):
    messages: Annotated[list, add]   # 自动合并 messages
    steps: int

# 假设有一个 LLM 封装和工具列表
# llm = ChatOpenAI(model="gpt-4o")
# tools = [search_tool, calculator_tool]

def agent_node(state: AgentState):
    # 每次调用 LLM，让模型决定下一步动作（或给出最终答案）
    response = llm.invoke(state["messages"])
    return {"messages": [response], "steps": state.get("steps", 0) + 1}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    # 如果模型有 tool_calls，则继续调用工具；否则结束
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    return "end"

builder = StateGraph(AgentState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {
    "continue": "tools",
    "end": END
})
builder.add_edge("tools", "agent")  # 工具执行完回到 agent，形成循环

graph = builder.compile()
```

在这个例子中，`agent` 节点调用 LLM，LLM 可能返回工具调用请求。若 LLM 请求调用工具，`should_continue` 返回 `"continue"`，则进入 `tools` 节点执行工具；工具结果会被追加到 `messages` 中，然后重新回到 `agent` 节点。这就是“思考-行动-观察”闭环的完整实现。

### 状态的高效管理：reducers、状态合并与切片

在上述 `AgentState` 中，`messages` 使用了 `Annotated[list, add]`。这里的 `add` 是一个 reducer（归约器），它告诉 LangGraph：当节点返回新的 `messages` 时，不是直接覆盖，而是将新消息**追加**到已有列表中。

LangGraph 支持自定义 reducer。例如，希望合并两个字典而非覆盖：

```python
def merge_dict(left, right):
    merged = left.copy()
    merged.update(right)
    return merged

class CustomState(TypedDict):
    config: Annotated[dict, merge_dict]
```

你还可以对状态进行“切片”或裁剪，以避免消息列表无限增长。例如只保留最近 10 条消息：

```python
def trim_messages(state: AgentState):
    state["messages"] = state["messages"][-10:]
    return state
```

在节点之间显式管理消息长度，是控制 token 成本的关键手段。LangGraph 本身不会自动清理状态，因此需要开发者自行决定哪些信息需要丢弃、哪些需要保留。

### 持久化 / 记忆：使用 Checkpointer 保存和恢复会话

默认情况下，Graph 每次 `invoke` 都是无状态的，执行结束后状态便会消失。为了让 Agent 拥有“长期记忆”，LangGraph 提供了检查点机制：每次执行时，它会保存一份包含完整状态和执行位置的检查点。

最简单的内存检查点是 `MemorySaver`：

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "user-123"}}

# 第一次对话
graph.invoke({"messages": [{"role": "user", "content": "我叫小明"}]}, config)

# 第二次对话：Agent 能回忆起之前的信息
graph.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]}, config)
```

通过 `thread_id` 隔离不同会话。相同 `thread_id` 的多次 `invoke` 会共享同一个状态，从而实现多轮对话的长期记忆。

LangGraph 还支持其他持久化后端，如 `SqliteSaver`、`PostgresSaver`，以及 LangGraph Platform 提供的云存储。使用数据库检查点后，即使服务器重启，Agent 也能从任意一步恢复。

---

## 第四章：实战案例与最佳实践

### 案例一：构建带反思能力的问答 Agent

“反思”是一种让 AI 自我改进的技术。我们可以让 Agent 先生成一版答案，再对答案进行批评，然后根据批评修正答案。用 LangGraph 实现这一流程非常直观：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class ReflectionState(TypedDict):
    question: str
    draft: str
    critique: str
    final_answer: str

def generate_draft(state: ReflectionState):
    # 实际项目中调用 LLM
    return {"draft": f"这是一个关于“{state['question']}”的初步回答。"}

def critique_draft(state: ReflectionState):
    # 让另一个 LLM 审查草稿
    if "不够详细" in state["draft"]:
        return {"critique": "回答太简单，需要补充具体数据。"}
    return {"critique": "回答可以接受。"}

def refine_answer(state: ReflectionState):
    # 根据批评修改回答
    return {"final_answer": f"【修正版】{state['draft']}（补充了更多细节）"}

def should_refine(state: ReflectionState) -> str:
    if state["critique"].startswith("回答太简单"):
        return "refine"
    return "done"

builder = StateGraph(ReflectionState)
builder.add_node("generate_draft", generate_draft)
builder.add_node("critique_draft", critique_draft)
builder.add_node("refine_answer", refine_answer)

builder.add_edge(START, "generate_draft")
builder.add_edge("generate_draft", "critique_draft")
builder.add_conditional_edges("critique_draft", should_refine, {
    "refine": "refine_answer",
    "done": END
})
builder.add_edge("refine_answer", "critique_draft")  # 再次检查

graph = builder.compile()
```

在这个图中，“生成草稿 → 批评 → 修正 → 再批评”构成了循环，直到批评通过才结束。这种结构可用于生成高质量内容、代码审查、数学题求解等场景。

### 案例二：多步骤工作流（数据采集 → 分析 → 生成报告）

LangGraph 不仅适合 Agent 循环，也适合传统的批处理工作流。下面是一个数据分析管道的例子：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class ReportState(TypedDict):
    keyword: str
    raw_data: list
    analysis: dict
    report: str

def collect_data(state: ReportState):
    # 模拟数据采集
    raw_data = [{"title": "AI 新闻 1", "views": 1000},
                {"title": "AI 新闻 2", "views": 2000}]
    return {"raw_data": raw_data}

def analyze_data(state: ReportState):
    total_views = sum(item["views"] for item in state["raw_data"])
    return {"analysis": {"total_views": total_views, "count": len(state["raw_data"])}}

def write_report(state: ReportState):
    a = state["analysis"]
    report = f"共采集 {a['count']} 条数据，总阅读量 {a['total_views']}。"
    return {"report": report}

builder = StateGraph(ReportState)
builder.add_node("collect_data", collect_data)
builder.add_node("analyze_data", analyze_data)
builder.add_node("write_report", write_report)

builder.add_edge(START, "collect_data")
builder.add_edge("collect_data", "analyze_data")
builder.add_edge("analyze_data", "write_report")
builder.add_edge("write_report", END)

graph = builder.compile()
result = graph.invoke({"keyword": "人工智能"})
print(result["report"])
```

这个工作流与第一章的“输入 → 处理 → 输出”结构一致，只不过每个节点承担了更复杂的职责。LangGraph 的优势在于：你可以随时在任意两个节点之间插入 `send` 事件，或配置人工审核节点，而无需推翻现有代码。

### 常见坑与调试技巧

**1. 忘记返回状态字段**  
节点函数返回的所有 key 都会合并到全局 State。如果漏写了某个字段，后续节点就可能拿不到数据。建议在 State 定义中将必填字段设为必填，并使用 `get()` 方法安全取值。

**2. 循环没有终止条件**  
循环虽然强大，但也容易写出死循环。务必在条件边或节点函数中设置迭代上限。例如在 `AgentState` 中加入 `steps`，当 `steps > 5` 时强制结束：

```python
def should_continue(state):
    if state["steps"] >= 5:
        return "end"
    return "continue"
```

**3. 状态对象被意外修改**  
LangGraph 默认使用不可变状态语义？实际上它不会冻结对象，但最佳实践是：不要在节点内部直接修改传入的 `state`，而是返回新字典。这样可以避免难以追踪的副作用。

**4. 检查点与并发问题**  
使用同一 `thread_id` 并发调用 Graph 时，状态可能互相覆盖。如果情况复杂，可以为每次对话使用不同的 `thread_id`，或者通过 `invoke` 的 `config` 进行区分。

调试时，推荐使用 `graph.stream()` 而非 `invoke()`，它能逐节点打印中间状态：

```python
for event in graph.stream({"question": "什么是LangGraph?"}):
    for node_name, state in event.items():
        print(f"节点 {node_name} 更新了: {state}")
```

### 最佳实践总结

- **State 结构设计原则**：State 是节点间通信的唯一通道。字段不宜过多，保持扁平化；将“数据”（如 `raw_data`）和“元数据”（如 `steps`、`status`）分开；敏感信息不要放在 State 中持久化。
- **节点职责划分**：每个节点只做一件事。一个节点做 `collect_data`，另一个做 `analyze_data`，而不是混在一起。这样便于复用、测试和替换。
- **错误处理与超时控制**：在节点函数内部捕获异常并放入 State 的 `error` 字段，然后用条件边进入错误处理节点。对外部 API 调用，务必设置超时和重试逻辑。

```python
def safe_node(state):
    try:
        result = call_external_api()
        return {"ok": True, "data": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def route_on_error(state):
    if not state["ok"]:
        return "retry_or_fallback"
    return "next_step"
```

### 学习资源推荐

如果你希望继续深入，以下资源值得关注：

- **LangGraph 官方文档**：https://langchain-ai.github.io/langgraph/  
  这里有最权威的概念解释、API 参考和示例。
- **LangGraph GitHub 仓库**：https://github.com/langchain-ai/langgraph  
  源码和示例代码是最好的学习材料。
- **LangChain 官方课程 / DeepLearning.AI 课程**：其中包含专门的 LangGraph 主题课程，适合视频学习。
- **社区博客与案例**：Medium、微信公众号上有很多实战分享，搜索“LangGraph 教程”即可。

---

## 结语：从简单到进阶，构建真正可控的 AI 应用

LangGraph 不是又一个“大模型封装库”，而是一个能让你精确控制 AI 应用行为的编排引擎。它把复杂的状态管理、循环控制和人机协同问题，转化为清晰、可测试的图结构。无论你是在构建带反思能力的问答 Agent，还是搭建多步骤数据分析管道，LangGraph 都能让你告别“所有逻辑堆在一个函数里”的泥潭。

希望这篇入门指南能帮助你建立对 LangGraph 的整体认知。从最小的“输入 → 处理 → 输出” Graph 开始，到条件分支、循环、持久化，再到完整的实战案例——你会发现，可控的 AI 应用其实并不遥远。现在，就去构建你的第一个 LangGraph 吧！