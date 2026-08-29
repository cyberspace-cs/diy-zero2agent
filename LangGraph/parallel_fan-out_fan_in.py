from typing import TypedDict, Annotated, List, Optional
import operator
from langgraph.graph import StateGraph, END

class ParallelState(TypedDict):
    input: str
    results: Annotated[List[str],operator.add]

# ==== State ====
class CricketState(TypedDict, total=False):
    player_name: str
    batting_avg: float
    bowling_avg: float
    fielding_rating: float
    analyses: Annotated[List[str],operator.add] #三个并行节点的输出
    final_report: str
    task_type: str # 任务类型，"simple"或"complex"

# ==== 并行节点 ====
def analyze_batting(state: CricketState) -> dict:
    """分析击球表现"""
    avg = state["batting_avg"]
    if avg>=50:
        level = "世界级"
    elif avg>=35:
        level = "优秀"
    elif avg>=20:
        level = "一般"
    else:
        level = "较弱"

    analysis = f"[击球] 平均分 {avg},评级：{level}"
    return { "analyses":[analysis]}

def analyze_bowling(state: CricketState) -> dict:
    """分析投球表现(投球均值越低越好)"""
    avg = state["bowling_avg"]
    if avg<=20:
        level = "世界级"
    elif avg<=30:
        level = "优秀"
    elif avg<=40:
        level = "一般"
    else:
        level = "较弱"

    analysis = f"[投球] 平均分 {avg},评级：{level}"
    return { "analyses":[analysis]}

def analyze_fielding(state: CricketState) -> dict:
    """分析防守表现"""
    rating = state["fielding_rating"]
    if rating>=8:
        level = "出色"
    elif rating>=6:
        level = "良好"
    elif rating>=4:
        level = "一般"
    else:
        level = "需改进"

    analysis = f"[守场] 评分 {rating}/10,评级：{level}"
    return { "analyses":[analysis]}


# ==== 汇聚节点 ====

def aggregate_results(state: CricketState) -> dict:
    """汇聚三个分析结果，生成总报告"""
    name = state["player_name"]
    analyses = state["analyses"]
    task_type = state.get("task_type") or "simple"
    if task_type == "simple":
        report = f"=== {name} 简单评估报告 ===\n"
        
        for a in analyses:
            report += f"  {a}\n"
        score = min(state["batting_avg"] / 50, 1.0) * 100 
        report += f"简单评估得分：{score:.1f} / 100"
    else:
            
        report = f"=== {name} 综合评估报告 ===\n"
        for a in analyses:
            report += f"  {a}\n"

        # 计算综合评分（简化逻辑）
        score = (
            min(state["batting_avg"] / 50, 1.0) * 40 +
            max(0, (40 - state["bowling_avg"]) / 40) * 40 +
            state["fielding_rating"] / 10 * 20
        )
        report += f"综合评估得分：{score:.1f} / 100"

    return {"final_report": report}

 
# ==== 入口节点 ====

def start_node(state: CricketState) -> dict:
    """入口节点，什么都不做，只是作为Fan-out的起点"""
    return {}

# # ==== 构建状态图 ====

# graph = StateGraph(CricketState)

# graph.add_node("start", start_node)
# graph.add_node("analyze_batting", analyze_batting)
# graph.add_node("analyze_bowling", analyze_bowling)
# graph.add_node("analyze_fielding", analyze_fielding)
# graph.add_node("aggregate_results", aggregate_results)

# graph.set_entry_point("start")

# # Fan-out 并行节点
# graph.add_edge("start", "analyze_batting")
# graph.add_edge("start", "analyze_bowling")
# graph.add_edge("start", "analyze_fielding")

# # Fan-in 汇聚节点
# graph.add_edge("analyze_batting", "aggregate_results")
# graph.add_edge("analyze_bowling", "aggregate_results")
# graph.add_edge("analyze_fielding", "aggregate_results")

# graph.add_edge("aggregate_results", END)

# app = graph.compile()

# # 运行状态图
# result = app.invoke({
#     "player_name": "Rohit Sharma",
#     "batting_avg": 59.8,
#     "bowling_avg": 34.0,
#     "fielding_rating": 9.0,
#     "analyses": [],
#     "final_report": "",
# })

# print(result["final_report"])

def route_by_type(state: CricketState) -> dict:
    """根据任务类型返回路由 key,默认为simple"""
    return state.get("task_type") or "simple"

# ==== 构建状态图 ====


graph = StateGraph(CricketState)


graph.add_node("start", start_node)

graph.add_node("quick_analysis", start_node)
graph.add_node("detailed_analysis", start_node)

graph.add_node("analyze_batting", analyze_batting)
graph.add_node("analyze_bowling", analyze_bowling)
graph.add_node("analyze_fielding", analyze_fielding)
graph.add_node("aggregate_results", aggregate_results)

graph.set_entry_point("start")


#入口节点后面：根据任务类型分流
graph.add_conditional_edges(
    "start",
    route_by_type,
    {
        "simple": "quick_analysis",
        "complex": "detailed_analysis",
    }
)

# Fan-out 并行节点
graph.add_edge("quick_analysis", "analyze_batting")


graph.add_edge("detailed_analysis", "analyze_batting")
graph.add_edge("detailed_analysis", "analyze_bowling")
graph.add_edge("detailed_analysis", "analyze_fielding")

# Fan-in 汇聚节点
graph.add_edge("analyze_batting", "aggregate_results")
graph.add_edge("analyze_bowling", "aggregate_results")
graph.add_edge("analyze_fielding", "aggregate_results")

graph.add_edge("aggregate_results", END)

app = graph.compile()

# 运行状态图
result = app.invoke({
    "player_name": "Rohit Sharma",
    "batting_avg": 62.8,
    "bowling_avg": 10.0,
    "fielding_rating": 10.0,
    "analyses": [],
    "final_report": "",
})

print(result["final_report"])

# 运行状态图
result = app.invoke({
    "player_name": "Rohit Sharma",
    "batting_avg": 62.8,
    "bowling_avg": 10.0,
    "fielding_rating": 10.0,
    "analyses": [],
    "final_report": "",
    "task_type": "complex",
})

print(result["final_report"])

# 运行状态图
result = app.invoke({
    "player_name": "Virat Kohli",
    "batting_avg": 49.8,
    "bowling_avg": 9.0,
    "fielding_rating": 9.0,
    "analyses": [],
    "final_report": "",
})

print(result["final_report"])

# 运行状态图
result = app.invoke({
    "player_name": "Virat Kohli",
    "batting_avg": 49.8,
    "bowling_avg": 9.0,
    "fielding_rating": 9.0,
    "analyses": [],
    "final_report": "",
    "task_type": "complex",
})

print(result["final_report"])
