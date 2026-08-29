from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

# 顺序图：第一个可运行的Workflow

# 1. 定义 State
# 初始 State 怎么设置
# 注意调用 invoke 时我们传入了完整的初始 state，包括 bmi: 0.0、category: ""、report: "" 这些“空值”。

# 这是因为 TypedDict 要求字段完整。实际项目里通常有两种处理方式：

# 方式一：用 Optional 加默认值
# 方式二：用 total=False 允许字段不全为空
class BMIState(TypedDict, total=False):
    name: str
    height_cm: float
    weight_kg: float
    bmi: Optional[float]
    category: Optional[str]
    advice: Optional[str]
    report: Optional[str]
    error: Optional[str]

# 2. 定义节点
def validate_input(state: BMIState) -> dict:
    """验证输入数据"""
    height=state['height_cm']
    weight=state['weight_kg']
    if height<=0 or height>300:
        return {"error":f"身高数据异常:{height}cm"}

    if weight<=0 or weight>500:
        return {"error":f"体重数据异常:{weight}kg"}

    return {"error":None}

def calculate_bmi(state: BMIState) -> dict:
    """计算 BMI"""
    if state.get("error"):
        return {}
    height_m = state["height_cm"] / 100
    bmi = state["weight_kg"] / (height_m ** 2)
    return {"bmi": round(bmi, 2)}

def classify_bmi(state: BMIState) -> dict:
    """分类 BMI"""
    if state.get("error"):
        return {}
    bmi = state["bmi"]
    if bmi < 18.5:
        category = "偏瘦"
    elif 18.5 <= bmi < 24.9:
        category = "正常"
    elif 24.9 <= bmi < 29.9:
        category = "超重"
    else:
        category = "肥胖"
    return {"category": category}

def generate_advice(state: BMIState) -> dict:
    """生成建议"""
    if state.get("error"):
        return {}
    category = state["category"]
    advice_map = {
        "偏瘦": "建议适量增加营养摄入，加强力量训练，必要时咨询营养师。",
        "正常": "保持当前的饮食和运动习惯，定期体检。",
        "超重": "建议控制饮食，每周至少进行 150 分钟中等强度有氧运动。",
        "肥胖": "建议在医生指导下制定减重计划，注意饮食结构和规律运动。"
    }

    return {"advice": advice_map[category]}

def format_report(state: BMIState) -> dict:
    """格式化最终报告"""
    if state.get("error"):
        report = f"错误：{state['error']}"
    else:
        report = f"""
=== BMI 健康报告  ===
姓名：{state['name']}
身高：{state['height_cm']} cm      
体重：{state['weight_kg']} kg
BMI：{state['bmi']}
分类：{state['category']}
建议：{state['advice']}
""".strip()

    return {"report": report}

# 3. 构建状态图
graph=StateGraph(BMIState)

# 添加节点
graph.add_node("validate", validate_input)
graph.add_node("calculate", calculate_bmi)
graph.add_node("classify", classify_bmi)
graph.add_node("advise", generate_advice)
graph.add_node("format", format_report)

# 设置顺序边
graph.set_entry_point("validate")
graph.add_edge("validate", "calculate")
graph.add_edge("calculate", "classify")
graph.add_edge("classify", "advise")
graph.add_edge("advise", "format")
graph.add_edge("format", END)

app=graph.compile()

#运行
result = app.invoke({
    "name": "张三",
    "height_cm": 175.0,
    "weight_kg": 70.0,
    # "bmi": 0.0,
    # "category": "",
    # "advice": "",
    # "report": "",
    # "error": None
})

print(result["report"])

# 用stream观察每个节点的输出
for step in app.stream({
    "name": "王五",
    "height_cm": 160.0,
    "weight_kg": 80.0,
    "bmi": 0.0,
    "category": "",
    "advice": "",
    "report": "",
    "error": None
}):
    node_name, state_update = list(step.items())[0]
    print(f"节点: {node_name}, 输出: {state_update}")
