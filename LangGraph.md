# ClinicalMind LangGraph 工作流详解

## 1. 架构概览

ClinicalMind 使用 LangGraph 的 `StateGraph` 构建整个多 Agent 医疗问诊工作流。所有节点共享一个 `ClinicalState`，通过声明式的节点和条件边编排流程，用 `interrupt()` 实现人机交互。

### 文件结构

```
clinicalmind_lg/
├── state.py      # ClinicalState 定义 + initial_state 工厂
├── prompts.py    # 7 个中文 System Prompt
├── llm.py        # LLM 客户端（延迟加载，兼容 5 个 Provider）
├── nodes.py      # 6 个节点 + 2 个路由函数
├── graph.py      # StateGraph 构建 + MemorySaver 编译
└── console.py    # 交互式控制台（interrupt 消费端）
```

---

## 2. ClinicalState — 共享状态

所有节点读取和写入同一个 `ClinicalState` TypedDict。LangGraph 根据字段上的 `Annotated` reducer 自动合并节点输出。

```python
class ClinicalState(TypedDict):
    # 用户输入
    chief_complaint: str                     # 主诉
    current_user_input: str                  # 当前消息

    # 问诊状态（带 reducer 自动合并）
    collected_info: Annotated[dict, merge]   # {qid: extracted_value}
    raw_answers: Annotated[dict, merge]      # {qid: raw_answer}
    asked_questions: Annotated[list, append] # 已问 ID 列表
    interview_round: int                     # 当前轮次

    # 临床推理
    differential_diagnoses: list[dict]       # 鉴别诊断演变
    red_flags: Annotated[list, append]       # 危险信号积累

    # 安全分诊
    is_emergency: bool                       # 急诊标记
    triage_flags: list[str]

    # 路由
    intent: str                              # diagnosis|research|planning
    route: str                               # 实际路由目标

    # 产出
    diagnosis_report: dict                   # 结构化诊断
    treatment_plan: dict                     # 治疗计划
    research_answer: str                     # 知识问答结果

    # 流程控制
    phase: str                               # classify|interviewing|diagnosing|completed
    action: str                              # ask|synthesize|end
```

### Reducer 机制

```python
# 列表追加 — red_flags 随时间积累
red_flags: Annotated[list[str], lambda a, b: a + b]

# 字典合并 — 后写入覆盖前写入
collected_info: Annotated[dict[str, str], lambda a, b: {**a, **b}]
```

---

## 3. Graph 拓扑

```
                    START
                      │
                      ▼
                ┏━━━━━━━━━┓
                ┃  entry  ┃  安全分诊(规则+LLM) + 意图分类(LLM)
                ┗━━━┳━━━━━┛
                    │
                    ▼
             route_after_entry
                    │
    ┌───────────────┼──────────────────┐
    ▼               ▼                  ▼
interview_loop   research    standalone_planning
    │               │                  │
    │ (内部循环)     │                  │
    │               ▼                  ▼
    │             END                END
    │
    ▼
  diagnose ──────▶ treatment_plan ──▶ END
```

### 节点表

| 节点 | 职责 | LLM 调用 |
|------|------|----------|
| `entry` | 关键词安全分诊 → LLM 深度检查 → LLM 意图分类 → 设定 route | 2 次 |
| `interview_loop` | 生成问题 → `interrupt()` 等待 → 处理答案 → 判断是否充分 | 2 次/轮 |
| `diagnose` | 汇总信息 → LLM 生成结构化诊断报告 | 1 次 |
| `treatment_plan` | 基于诊断 → LLM 生成治疗计划 | 1 次 |
| `research` | 直接回答医学知识问题 | 1 次 |
| `standalone_planning` | 从用户文本直接生成治疗计划 | 1 次 |

### 边表

| 起点 | 终点 | 条件 |
|------|------|------|
| `START` | `entry` | 无 |
| `entry` | `interview_loop` | route=diagnosis |
| `entry` | `research` | route=research |
| `entry` | `standalone_planning` | route=planning |
| `entry` | `diagnose` | is_emergency=True |
| `interview_loop` | `diagnose` | 信息充分/达最大轮次 |
| `diagnose` | `treatment_plan` | 非急诊 |
| `diagnose` | `END` | 急诊 |
| `treatment_plan` | `END` | 无 |
| `research` | `END` | 无 |
| `standalone_planning` | `END` | 无 |

---

## 4. 节点详解

### 4.1 entry_node

```
用户输入
    │
    ▼
[规则分诊]  6组关键词 (胸痛+呼吸/神经/出血/过敏)
    ├─ 命中 → is_emergency=True → route=diagnose
    │
    └─ 未命中 → LLM深度安检 → LLM意图分类
                    │
                    ▼
              diagnosis → interview_loop
              research → research
              planning → standalone_planning
```

- **规则检查**（毫秒级）：胸痛+呼吸困难、卒中表现、大出血、严重过敏
- **LLM 检查**：对边界案例进行深度分析
- **意图分类**：MasterAgent 将用户输入归类为 diagnosis/research/planning

### 4.2 interview_loop_node（核心）

一个函数替代了旧版 5 个类（Track1Agent、Track2Agent、InterviewOrchestrator、DynamicInterviewEngine、process_answer）。

```
for round in 0..12:
    ① 构建 Prompt（主诉 + 已收集信息 + 鉴别诊断 + 状态）
    ② LLM 生成问题 (INTERVIEW_SYSTEM_PROMPT)
       返回 {action: "ask"|"synthesize", questions: [...], diffs: [...]}
    ③ 合并鉴别诊断（按 diagnosis 名称去重）
    ④ 检查 synthesize：
        - action=="synthesize" AND meaningful>=3 → return
        - 无问题 AND meaningful>=5 → return
    ⑤ interrupt({type: "ask_questions", questions, diffs, round, ...})
        → 暂停等待用户输入
    ⑥ 恢复后处理答案：
        - "||" 分隔多个答案
        - 数字选项映射到文本
        - LLM 提取关键信息 (ANSWER_PROCESSOR_PROMPT)
        - 存入 collected_info / raw_answers

# 超最大轮次 → 强制进入诊断
```

**为什么是内部 for 循环而不是 LangGraph cycle？**

每次 `interrupt()` 后需要更新本地变量（`collected`, `diffs`），如果依赖 LangGraph 的 state merge 回环，会导致状态覆盖和排序问题。内部循环确保状态一致性。

### 4.3 diagnose_node

```
collected_info + raw_answers + diffs + red_flags
    │
    ▼
[急诊检查] → 跳过 LLM，返回急诊报告
    │
    ▼
[构建综合摘要] 主诉 + 结构化信息 + 鉴别诊断演变 + 原始描述 + 安全标记
    │
    ▼
[LLM] DIAGNOSIS_SYSTEM_PROMPT → {
    primary_diagnosis: "...",
    differential_diagnoses: [{diagnosis, icd11_code, reasoning}],
    confidence: "high|medium|low",
    severity: "mild|moderate|severe|emergency",
    key_findings: [...],
    recommended_tests: [...],
    recommended_actions: [...],
    red_flags: [...],
    follow_up_required: bool,
    follow_up_timeline: "...",
    disclaimer: "..."
}
```

### 4.4 treatment_plan_node

基于 `diagnosis_report` 生成结构化治疗计划：目标、药物讨论点、非药物治疗、护理建议、康复阶段、随访安排。

### 4.5 research_node / standalone_planning_node

快捷路径节点。research 直接回答医学知识问题；standalone_planning 从文本直接生成治疗计划（无需先诊断）。

---

## 5. 路由函数

### route_after_entry

```python
def route_after_entry(state):
    if state["is_emergency"]:
        return "diagnose"           # 急诊跳过问诊
    route = state.get("route", "diagnosis")
    return {
        "diagnosis": "interview_loop",
        "research": "research",
        "planning": "standalone_planning",
    }.get(route, "interview_loop")
```

### route_after_diagnosis

```python
def route_after_diagnosis(state):
    if state["is_emergency"]:
        return "__end__"             # 急诊不生成治疗计划
    return "treatment_plan"
```

---

## 6. interrupt() 人机交互机制

```python
# 在 interview_loop_node 内部
human_response = interrupt({
    "type": "ask_questions",
    "questions": [...],           # 1-2 道问题
    "differential_diagnoses": [...],
    "red_flags": [...],
    "round": N,
    "collected_count": N,
})

# ↑ LangGraph 在此暂停，MemorySaver 保存 checkpoint
# ↓ 用户回答后，Console 调用 Command(resume=...) 恢复
```

### Console 消费端循环

```python
config = {"configurable": {"thread_id": uuid4()}}
state = initial_state(chief_complaint)

while True:
    result = await graph.ainvoke(state, config)
    gs = graph.get_state(config)

    if not gs.next and not gs.interrupts:
        return result  # 完成

    for it in gs.interrupts:
        if it.value["type"] == "ask_questions":
            # 展示问题、收集答案
            response = ask_questions(it.value["questions"])
            # 恢复执行
            state = Command(resume=response)
```

### 多问题协议

多个答案用 `||` 分隔：
```
用户输入 "2||1,3"
→ Q0: 选项[2] → "慢慢加重"
→ Q1: 选项[1,3] → "怕光, 怕声音"
```

---

## 7. 完整执行时序

```
 T0   用户输入 "头痛一周"
 T1   entry_node: 规则分诊(未命中) → LLM安全(无急诊) → LLM意图(diagnosis)
 T2   route_after_entry → interview_loop
 T3   interview_loop round=1: LLM 生成 2 道题 → interrupt()
      → Console 展示问题，用户回答 "2||1"
 T4   Command(resume="2||1") → 处理答案 → round=2
 ...  重复 3-12 轮 ...
 Ta   interview_loop return (phase=diagnosing)
 Tb   diagnose_node: LLM 诊断 → 紧张型头痛 (ICD-11: 8A80)
 Tc   treatment_plan_node: LLM 生成治疗计划
 Td   END → Console 展示结果
```

---

## 8. LLM 调用汇总

| 节点 | 单次调用数 | 模型配置 |
|------|-----------|----------|
| entry | 2 | `llm_fast` (512 tokens) |
| interview_loop | 2×N 轮 | `llm` (2048) + `llm_fast` |
| diagnose | 1 | `llm` (4096) |
| treatment_plan | 1 | `llm` (2048) |
| research | 1 | `llm` (2048) |

典型 6 轮问诊流程约 **18 次 LLM 调用**。

---

## 9. 扩展指南

### 添加新节点

```python
# nodes.py
async def drug_interaction_node(state: ClinicalState) -> dict[str, Any]:
    meds = state.get("diagnosis_report", {}).get("medications", [])
    # ... LLM 药物相互作用检查 ...
    return {"drug_report": {...}}

# graph.py
workflow.add_node("drug_check", drug_interaction_node)
workflow.add_conditional_edges("diagnose", ..., {"drug_check": "drug_check", ...})
workflow.add_edge("drug_check", "treatment_plan")
```

### 持久化到 SQLite

```python
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string("clinicalmind.db")
workflow.compile(checkpointer=memory)
# 中断后重启，相同 thread_id 恢复会话
```

### 添加 LangChain Tool

```python
from langchain_core.tools import tool

@tool
def search_guidelines(query: str) -> str:
    """检索临床指南"""
    ...  # 调用 SearXNG / 向量检索

# 在 interview_loop_node 中绑定
llm_with_tools = llm.bind_tools([search_guidelines])
```

---

## 10. 支持的 LLM Provider

| Provider | 环境变量 | Base URL |
|----------|---------|----------|
| DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com/v1` |
| OpenAI | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| 智谱 GLM | `GLM_API_KEY` | `https://open.bigmodel.cn/api/paas/v4` |
| Moonshot | `MOONSHOT_API_KEY` | `https://api.moonshot.cn/v1` |
| 阿里百炼 | `DASHSCOPE_API_KEY` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

配置方式：在 `.env` 中填写任意一个 Provider 的 API Key，启动时自动检测并创建客户端。
