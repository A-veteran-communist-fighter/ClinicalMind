# ClinicalMind LangGraph 工作流详解

## 1. 架构概览

ClinicalMind 使用 LangGraph 的 `StateGraph` 构建整个多 Agent 医疗问诊工作流。所有节点共享一个 `ClinicalState`，通过声明式的节点和条件边编排流程。

### 文件结构

```
clinicalmind_lg/
├── state.py      # ClinicalState 定义 + initial_state 工厂
├── prompts.py    # 8 个中文 System Prompt（含化验单解析）
├── llm.py        # LLM 客户端（延迟加载，兼容 5 个 Provider + 独立视觉模型）
├── nodes.py      # 7 个节点 + 化验单解析 + 3 个路由函数
├── graph.py      # StateGraph 构建 + MemorySaver 编译
└── console.py    # 交互式控制台
```

### 人机交互方式：状态驱动

```
1. interview_generate_node 生成问题 → 写入 state["current_questions"] → graph 在 END 暂停
2. console 读取 state["current_questions"] 展示给用户
3. 用户回答后，console 将答案写入 state["human_response"]，重新调用 graph.ainvoke()
4. graph 从 entry 进入，路由到 interview_process_node 处理答案
5. 处理后根据 phase 决定：继续生成问题 或 进入诊断
```

---

## 2. ClinicalState — 共享状态

```python
class ClinicalState(TypedDict):
    # 用户输入
    chief_complaint: str
    current_user_input: str

    # 问诊状态
    collected_info: Annotated[dict, merge]
    raw_answers: Annotated[dict, merge]
    asked_questions: Annotated[list, append]
    current_questions: list[dict]
    interview_round: int

    # 临床推理
    differential_diagnoses: list[dict]
    red_flags: Annotated[list, append]

    # 多模态
    lab_reports: Annotated[list[dict], append]  # 化验单解析结果

    # 安全分诊
    is_emergency: bool

    # 路由
    intent: str     # diagnosis|research|planning
    route: str

    # 产出
    diagnosis_report: dict
    treatment_plan: dict
    research_answer: str

    # 流程控制
    phase: str      # classify|awaiting_human|interviewing|diagnosing|completed
    action: str     # ask|synthesize|end

    # 人机交互
    human_response: str
```

### Reducer 机制

```python
red_flags: Annotated[list[str], lambda a, b: a + b]          # 列表追加
collected_info: Annotated[dict[str, str], lambda a, b: {**a, **b}]  # 字典合并
lab_reports: Annotated[list[dict], lambda a, b: a + b]       # 化验单累积
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
interview_generate  research    standalone_planning
    │               │                  │
    ▼               ▼                  ▼
   END             END                END
    │
    │ (console 读取 questions, 用户回答 / lab <path> 上传化验单)
    │
    ▼
interview_process ←── entry（route_after_entry 检测到 awaiting_human+答案 → 路由至此）
    │
    ▼
route_after_process:
    ├─ interviewing → interview_generate  （继续问）
    └─ diagnosing → diagnose            （信息充分）

diagnose → treatment_plan → END
```

### 节点表

| 节点 | 职责 | LLM 调用 |
|------|------|----------|
| `entry` | 规则安全分诊 → LLM 深度检查 → LLM 意图分类 → 路由 | 2 次 |
| `interview_generate` | 构建 prompt → LLM 生成 1-2 个问题（含化验数据）→ END | 1 次 |
| `interview_process` | 解析用户答案 → LLM 提取关键信息 → 决定下一步 | 0-1 次 |
| `diagnose` | 汇总信息（含化验）→ LLM 结构化诊断 | 1 次 |
| `treatment_plan` | 基于诊断 → LLM 治疗计划 | 1 次 |
| `research` | 医学知识问答 | 1 次 |
| `standalone_planning` | 从文本直接生成治疗计划 | 1 次 |

独立函数（非 graph 节点）：

| 函数 | 职责 |
|------|------|
| `parse_lab_report_image` | 视觉模型识别化验单图片 → 结构化指标 |

### 边表

| 起点 | 终点 | 条件 |
|------|------|------|
| START | entry | 无 |
| entry | interview_generate | route=diagnosis，首次 |
| entry | interview_process | route=diagnosis，resume |
| entry | research | route=research |
| entry | standalone_planning | route=planning |
| entry | diagnose | is_emergency=True |
| interview_generate | END | 无（暂停） |
| interview_process | interview_generate | phase=interviewing |
| interview_process | diagnose | phase=diagnosing |
| diagnose | treatment_plan | 非急诊 |
| diagnose | END | 急诊 |
| treatment_plan | END | 无 |

---

## 4. 节点详解

### 4.1 entry_node

```
用户输入
    │
    ▼
[规则分诊]  正则匹配 (胸.{0,3}痛/突然.{0,5}痛 + 6组关键词)
    ├─ 命中 → is_emergency=True → route=diagnose
    │
    └─ 未命中 → LLM深度安检 → LLM意图分类
```

三层安全分诊：

| 层 | 方式 | 示例 |
|---|------|------|
| 精确关键词 | `"胸痛" in text` | 匹配"我胸痛" |
| 松散正则 | `胸.{0,3}痛` / `突然.{0,5}痛` | 匹配"我胸口突然很痛" |
| 紧迫性词 | `"突然"、"撕裂"、"难以忍受"` | 判断急诊级别 |

### 4.2 interview_generate_node

生成 1-2 个问诊问题。如果有化验数据，自动注入 prompt 让 LLM 在问诊时考虑化验结果。

### 4.3 interview_process_node

处理用户回答。支持 `lab` 命令——用户在回答问题时可以上传化验单，系统解析后重新生成问题。

### 4.4 化验单解析（parse_lab_report_image）

```
用户输入 lab <image_path>
    │
    ▼
读取图片 → base64 编码 → 视觉 LLM → 结构化 JSON
    │
    ▼
[
  {"indicator_name": "空腹血糖", "value": "8.2", "unit": "mmol/L",
   "abnormal": true, "abnormal_level": "moderate", "abnormal_direction": "high"},
  ...
]
    │
    ▼
注入 state["lab_reports"] → interview_generate prompt → diagnose summary
```

视觉模型独立于主 LLM，通过 `VISION_API_KEY` / `VISION_MODEL` / `VISION_BASE_URL` 单独配置。未配置时 `lab` 命令返回清晰的降级提示，不影响其他功能。

---

## 5. 路由函数

### route_after_entry

```python
def route_after_entry(state):
    if state["is_emergency"]:
        return "diagnose"
    if state["phase"] == "awaiting_human" and state["human_response"]:
        return "interview_process"  # resume 时处理答案
    route = state.get("route", "diagnosis")
    return {"diagnosis": "interview_generate", ...}.get(route)
```

### route_after_process

```python
def route_after_process(state):
    if state["phase"] == "diagnosing":
        return "diagnose"
    return "interview_generate"
```

### route_after_diagnosis

```python
def route_after_diagnosis(state):
    if state["is_emergency"]:
        return "__end__"
    return "treatment_plan"
```

---

## 6. Console 执行循环

```python
state = initial_state(chief_complaint)

for _ in range(MAX_ROUNDS):
    result = await graph.ainvoke(state, config)
    phase = result["phase"]

    if phase == "awaiting_human":
        questions = result["current_questions"]
        answer = await ask_questions(questions)  # 支持 lab 命令
        state = {"human_response": answer}
    elif phase in ("diagnosing", "diagnosed", "completed"):
        return result
```

---

## 7. LLM 调用汇总

| 节点 | 单次调用数 | 模型配置 |
|------|-----------|----------|
| entry | 2 | fast (512 tokens) |
| interview_generate | 1/轮 | main (2048) |
| interview_process | 0-1/轮 | fast (512) |
| diagnose | 1 | main (4096) |
| treatment_plan | 1 | main (2048) |
| parse_lab_report | 1/次 | vision (独立配置) |

典型 6 轮问诊约 20 次 LLM 调用。化验单解析额外 1 次视觉模型调用。

---

## 8. 支持的 Provider

| 用途 | 环境变量 |
|------|---------|
| 主 LLM | `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `GLM_API_KEY` / `MOONSHOT_API_KEY` / `DASHSCOPE_API_KEY` |
| 视觉模型 | `VISION_API_KEY` + `VISION_MODEL` + `VISION_BASE_URL`（独立于主 LLM，未配则降级） |

---

## 9. 扩展指南

### 添加新的视觉 Provider

在 `clinicalmind_lg/llm.py` 的 `get_vision_llm()` 中配置：

```python
vision_model = os.getenv("VISION_MODEL", "gpt-4o")
vision_url = os.getenv("VISION_BASE_URL", "https://api.openai.com/v1")
```

### 持久化到 SQLite

```python
from langgraph.checkpoint.sqlite import SqliteSaver
memory = SqliteSaver.from_conn_string("clinicalmind.db")
workflow.compile(checkpointer=memory)
```
