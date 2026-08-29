# LangGraph状态图编程：构建可控、弹性的AI工作流

在传统软件开发中，工作流往往被建模为一系列线性或树状的步骤。但随着大语言模型（LLM）和智能体的兴起，业务流程的复杂性与不确定性急剧上升——决策分支需要依赖实时上下文，执行路径需要支持循环与回退，异常处理需要精细地插入到任何环节。传统的“管道式”工作流已经难以满足这种动态需求。**LangGraph** 正是为了应对这一挑战而诞生的图式状态管理框架。它通过**状态图**（StateGraph）将工作流建模为一张有向图，用**节点**承载业务逻辑，用**边**描述流转逻辑，以显式的**状态管理**贯穿全过程，从而让AI应用具备高度可控性、可恢复性和可扩展性。本文将围绕这一主题，从概念到实战，系统阐述LangGraph状态图编程的核心思路与落地方法。

## 1. LangGraph状态图编程概述：从传统工作流到图式状态管理

传统工作流引擎（如Airflow、Argo）通常基于DAG（有向无环图）构建，它们擅长处理依赖明确、顺序固定的批处理任务。然而，在LLM应用场景中，工作流往往是“活”的：同一个请求可能因为用户意图或模型输出而走向完全不同的路径；一个步骤可能需要反复执行直到满足某个条件；某个节点失败后不能简单终止，而是需要回退到前置状态重试或人工介入。面向这种动态性，图式状态管理应运而生。

LangGraph的核心思想是：将整个工作流视为一张**状态图**。状态图本身是一个具有记忆功能的运行时容器，它保存了所有节点共享的**状态**（State）。节点（Node）是状态转换的函数，它们从状态中读取所需信息，执行计算，然后生成对状态的更新。边（Edge）则定义了节点的执行顺序以及状态的可能走向。与DAG不同，LangGraph允许图中的环存在——这意味着循环和回退不是通过外部调度器硬编码实现的，而是图本身的内在表达能力。

这种设计带来几个关键优势：

- **动态路径**：通过条件边（Conditional Edge），节点可以实时决定下一个执行节点，使工作流具备决策能力。
- **状态持久性**：状态被显式建模并贯穿整个图的执行过程，节点之间通过状态传递信息，无需全局数据库或隐式变量。
- **可恢复性**：由于状态是自包含的，工作流可以被暂停、序列化、恢复，甚至回滚到历史状态，极大地提升了容错能力。
- **可读性**：将业务逻辑抽象为节点和边，使得复杂的流程可以被可视化地审查和维护，降低了理解成本。

从开发者的视角，LangGraph的编写体验与普通Python函数类似：你只需要定义状态schema、实现节点函数、连接边，然后编译并执行图。这种从“流程驱动”到“状态驱动”的转变，正是LangGraph状态图编程与生俱来的魅力。

## 2. 核心要素解析：状态图、节点与边的建模方式

要掌握LangGraph，必须先理解三个核心要素：**状态图**（StateGraph）、**节点**（Node）和**边**（Edge）。它们共同构成了工作流的骨架与血肉。

### 2.1 状态图：工作流的容器

状态图是LangGraph中最重要的数据结构。它定义了工作流的类型、初始状态和全部拓扑结构。在LangGraph中，状态通常是一个TypedDict或dataclass，表示所有需要在节点间传递的变量。例如：

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph

class WorkflowState(TypedDict):
    query: str
    category: str
    attempts: int
    result: str
    errors: Annotated[List[str], operator.add]  # 使用Reducer合并更新
```

`TypedDict`定义了每个字段的类型。`Annotated`和注释中的reducer（如`operator.add`）用于指定状态更新的合并策略。例如，当多个节点尝试更新`errors`字段时，默认行为会将新值追加到列表中，而不是覆盖。这种机制让状态更新变得声明式且可预测。

然后创建一个状态图实例：

```python
graph = StateGraph(WorkflowState)
```

### 2.2 节点：执行动作的单元

节点是工作流中执行具体功能的函数。每个节点接收当前状态作为输入，返回一个`dict`，其中包含对状态的更新。LangGraph稍后会将返回值合并到全局状态中。

```python
def classify_node(state: WorkflowState):
    query = state["query"]
    category = classify(query)
    return {"category": category}
```

节点可以是普通函数，也可以是可调用对象，甚至是异步函数（用于异步场景）。节点内部几乎可以执行任何操作：调用LLM、数据库查询、外部API、人工审批等。节点的设计原则是“无副作用 + 幂等”，因为LangGraph支持状态回放（Replay），节点可能会被多次执行。

### 2.3 边：定义流转逻辑

边连接节点，决定了执行的顺序。LangGraph提供两种边：

- **普通边（Normal Edge）**：无条件地从节点A到节点B，表示A执行完后必定执行B。
- **条件边（Conditional Edge）**：从节点A出发，根据一个路由函数的结果，动态选择去往多个节点中的一个。

普通边的添加方式：

```python
graph.add_edge("classify_node", "gate_node")
```

条件边需要提供一个路由函数，该函数接收状态，返回下一个节点的名称：

```python
def route_after_classify(state: WorkflowState):
    if state["category"] == "refund":
        return "refund_handler"
    elif state["category"] == "tech_support":
        return "tech_support_handler"
    else:
        return "human_review"

graph.add_conditional_edges(
    "classify_node",
    route_after_classify,
    {
        "refund_handler": "refund_node",
        "tech_support_handler": "ts_node",
        "human_review": "human_node",
    }
)
```

条件边是LangGraph实现“动态流转”的关键武器。它让工作流不再是预设好的铁路轨道，而是一个智能的路径选择器。

### 2.4 图的编译与执行

定义好节点和边后，需要将图实例化并编译：

```python
graph.add_node("classify_node", classify_node)
# ... 添加其他节点和边
app = graph.compile()
```

执行图时，传入初始状态：

```python
initial_state = {"query": "我的订单被错误扣款了", "attempts": 0}
final_state = app.invoke(initial_state)
```

调用`invoke`后，LangGraph会从图的入口开始，沿着边反复执行节点，直到到达终点（出口）或无法继续。整个过程由内部状态机驱动，我们只需要维护好状态。

## 3. 状态管理实战：在LangGraph中实现动态流转与条件分支

这一节我们通过一个真实的客服工单处理场景，展示如何使用LangGraph实现动态流转与条件分支。场景要求：

- 接收用户的问题文本。
- 对问题进行分类（退款、技术支持、其他）。
- 根据分类进入不同的处理节点。
- 如果处理失败，允许重试；重试次数超过阈值，则转人工。
- 最终返回结果。

我们首先定义状态：

```python
class TicketState(TypedDict):
    user_message: str
    category: str
    attempts: int
    success: bool
    message: str
    result: str
```

接下来编写节点。第一个节点是分类器，它使用LLM或简单的规则将用户消息分类：

```python
def classify_ticket(state: TicketState):
    msg = state["user_message"]
    # 简化模拟分类逻辑
    if "退款" in msg:
        category = "refund"
    elif "网络" in msg or "崩溃" in msg:
        category = "tech_support"
    else:
        category = "other"
    return {"category": category}
```

然后是退款处理节点和技术支持节点，它们会尝试“解决”问题，并标记是否成功。为了模拟失败和重试，我们设置一个随机因子：

```python
import random

def refund_handler(state: TicketState):
    # 模拟尝试，成功率70%
    success = random.random() > 0.3
    result = "退款处理完成" if success else "退款处理失败：银行接口超时"
    return {
        "success": success,
        "result": result,
        "attempts": state.get("attempts", 0) + 1
    }

def tech_support_handler(state: TicketState):
    # 模拟技术支持，成功率80%
    success = random.random() > 0.2
    result = "技术支持已提供解决方案" if success else "技术问题无法在线解决"
    return {
        "success": success,
        "result": result,
        "attempts": state.get("attempts", 0) + 1
    }
```

当分类为“other”时，直接转人工：

```python
def human_review(state: TicketState):
    return {"result": "转交人工客服处理", "success": True}
```

现在，我们需要条件路由来决定图的流向。首先，从分类节点到不同的处理节点：

```python
def route_after_classify(state: TicketState):
    return state["category"]  # "refund" -> "refund_handler", etc.
```

其次，从处理节点出发，我们需要判断是否成功，是否需要重试，或者已经达到最大重试次数。这个路由函数模拟了循环回退和异常分支：

```python
MAX_ATTEMPTS = 3

def route_after_handler(state: TicketState):
    if state["success"]:
        return "finish"
    elif state["attempts"] >= MAX_ATTEMPTS:
        return "human_review"  # 超出最大尝试，转人工
    else:
        return "retry"  # 返回到原处理节点重试
```

注意，返回`"retry"`时，我们想让流程回到刚才的处理节点。LangGraph的边支持指向任意节点，包括自身，因此我们可以构造一个环。为了简化，我们为“retry”创建一条指向处理节点的边：但不同分类对应的处理节点不同，所以这里更合适的做法是：在route_after_handler中根据`state["category"]`返回具体的处理节点，让环回到该节点。

```python
def route_after_handler(state: TicketState):
    if state["success"]:
        return "finish"
    elif state["attempts"] >= MAX_ATTEMPTS:
        return "human_review"
    else:
        # 重试：回到当前分类对应的处理节点
        return state["category"] + "_handler"
```

现在搭建图：

```python
from langgraph.graph import END

graph = StateGraph(TicketState)

graph.add_node("classify", classify_ticket)
graph.add_node("refund_handler", refund_handler)
graph.add_node("tech_support_handler", tech_support_handler)
graph.add_node("human_review", human_review)

graph.set_entry_point("classify")

graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {
        "refund": "refund_handler",
        "tech_support": "tech_support_handler",
        "other": "human_review",
    }
)

graph.add_conditional_edges(
    "refund_handler",
    route_after_handler,
    {
        "finish": END,
        "human_review": "human_review",
        "refund_handler": "refund_handler",  # 触发重试
    }
)

graph.add_conditional_edges(
    "tech_support_handler",
    route_after_handler,
    {
        "finish": END,
        "human_review": "human_review",
        "tech_support_handler": "tech_support_handler",
    }
)

graph.add_edge("human_review", END)

app = graph.compile()
```

执行：

```python
result = app.invoke({
    "user_message": "我的订单被重复扣款了，怎么办？",
    "attempts": 0,
})
print(result["result"], "尝试次数:", result["attempts"])
```

在这个实战例子中，我们充分利用了状态管理来实现动态流转：

- **条件分支**：`classify`节点后根据分类路由；`handler`后根据成功状态路由。
- **循环回退**：当处理节点失败且未超过最大次数时，边指向自身，形成循环，每次循环`attempts`递增。
- **状态更新**：每个节点通过返回dict更新状态，`attempts`的递增由`+`运算在node内部实现，也可以利用reducer自动合并。
- **终态判断**：成功或转人工后走向`END`。

这种模式完美展示了LangGraph如何以图结构天然表达“尝试→失败→重试→升级”的业务逻辑，而无需外部while循环或状态机代码。

## 4. 工作流编排进阶：多节点协作、循环回退与异常处理策略

实际生产环境中的工作流远不止一个简单的分支循环。我们需要应对更复杂的场景：多个节点并行处理不同的子任务、深度循环之间的状态协调、以及异常发生时如何优雅地降级或回滚。LangGraph提供了若干高级编排能力。

### 4.1 多节点协作：扇出与扇入

LangGraph支持通过`Send` API动态扇出（fan-out）多个并行节点，然后将结果汇聚到后续节点。例如，一个数据分析工作流需要分别查询多个数据源，然后汇总结果。我们可以用一个“分配节点”创建多个`Send`对象，每个对象指定目标节点和要传递的状态片段。LangGraph会并行执行这些节点，并把它们的结果合并回主状态。

```python
from langgraph.types import Send

def assign_tasks(state: MainState):
    tasks = state["tasks"]
    return [
        Send("process_task", {"task": t, "task_id": idx})
        for idx, t in enumerate(tasks)
    ]

graph.add_node("assign_tasks", assign_tasks)
graph.add_node("process_task", process_task)
graph.add_edge("assign_tasks", "process_task")
```

扇出后，所有`process_task`节点运行完毕，它们的输出会合并到主状态。合并策略由状态schema中对应字段的reducer决定。例如，我们可以用`operator.add`将多个返回的错误信息累积到`errors`列表中。

扇入（fan-in）通常在语言上是一个自动汇聚点——当所有并行分支都到达同一个节点时，该节点会看到合并后的完整状态。这种模式非常适用于Map-Reduce类工作流。

### 4.2 循环回退与嵌套状态

除了简单的自环，有时工作流需要“回退”到更早的节点，重新利用历史状态。例如，在LLM Agent的执行过程中，如果一次工具调用返回错误，Agent可能需要回到“计划”节点重新制定策略。LangGraph支持通过条件边将流程带到任何已定义的节点，形成有环图。

为了管理更复杂的回退，状态中需要保存足够的上下文。我们可以在状态中增加一个`history`字段，记录各个节点的输出，以便回退时重新读取。同时，通过reducer对`history`做累积，避免覆盖。

```python
class AgentState(TypedDict):
    task: str
    plan: List[str]
    history: Annotated[List[dict], operator.add]
    error: str
```

回退时，路由函数判断如果`error`发生，则返回 `"plan"` 节点，重新规划。规划节点会基于`history`调整计划，从而形成智能的循环。

### 4.3 异常处理策略

在LangGraph中，节点的异常可以直接被捕捉，也可以利用条件边转交到专门的异常处理节点。LangGraph允许为节点设置`retry_policy`或定义`try`/`except`语义。例如，使用`@graph.node`装饰器配合`retry`参数实现指数退避重试：

```python
from langgraph.graph import add_node, retry_policy

@add_node(graph, retry_policy=retry_policy(max_attempts=2, backoff_factor=2.0))
def fragile_node(state):
    # 可能抛出异常的网络调用
    ...
```

如果某个节点抛出了未捕获的异常，LangGraph默认会将异常传播到顶层，但我们也可以通过给该节点的边添加**错误路由**，让异常状态变成一个“特殊分支”。例如：

```python
def route_on_error(state):
    if state.get("error"):
        return "error_handler"
    return "next_ok"

graph.add_conditional_edges(
    "fragile_node",
    route_on_error,
    {"error_handler": "error_handler", "next_ok": "next_node"}
)
```

在节点内部，我们可以用try/except捕获异常，将错误信息写入状态，然后返回一个特殊标记，让路由函数决定下一步。这种基于状态流的异常处理比硬编码的try-catch更灵活，因为异常本身成为了工作流的一个状态，可以被记录、回溯，甚至触发人工审批。

**进阶技巧**：一个完备的工作流设计，往往会在图的最后添加一个总览节点，检查整个执行轨迹，如果发现整体失败，可以回到最初的入口重新执行或者启动补偿流程（Saga模式）。这完全可以通过LangGraph的条件边和状态累积实现。

## 结语：图式状态管理的未来

LangGraph状态图编程不仅仅是一个工具，更是一种思维方式的转变。它将工作流从静态的“步骤清单”升级为动态的“状态空间”，让AI系统能够像人类一样在复杂环境中进行推理、试错和协作。通过清晰的状态schema、可组合的节点、灵活的条件边，我们得以构建出更健壮、更透明的智能工作流。

随着LLM应用的场景不断拓展，状态图编程将在多Agent协作、自动规划、人机交互等前沿领域发挥越来越核心的作用。掌握LangGraph，不仅是掌握一个库，更是掌握了一套构建复杂AI系统的元语言。当你下次设计一个需要动态决策、循环回退和异常恢复的系统时，不妨尝试将业务逻辑画成一张图——也许，你就找到了最佳答案。