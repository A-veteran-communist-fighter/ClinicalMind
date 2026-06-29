# ClinicalMind

> 基于 LangGraph 的多智能体医学问诊系统

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/framework-LangGraph-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](./LICENSE)

ClinicalMind 是一个面向医学问诊场景的 **多 Agent 协作诊疗系统**。用户用自然语言描述症状，系统通过 LLM 驱动的动态问诊、安全分诊、鉴别诊断推理和治疗计划生成，输出结构化的循证诊断报告。

**⚠️ 医疗免责声明：本系统输出仅供参考，不能替代专业医疗诊断。如有胸痛、呼吸困难、意识障碍等急危重症，请立即就医。**

---

## 它做什么？

```
用户: "我最近一周持续头痛，额头和太阳穴胀痛，有时恶心，睡眠不好"

系统:
  1. [安全分诊] 规则 + LLM 双重检查 → 非急诊，继续
  2. [意图分类] LLM 识别 → diagnosis（诊断模式）
  3. [动态问诊] 生成问题 → 收集答案 → 更新鉴别诊断 → 循环
     Q1: "您头痛是突然开始还是慢慢加重的？"
     Q2: "头痛是持续性胀痛还是阵发性跳痛？"
     Q3: "有没有怕光、怕声音的情况？"
     ... (信息充分后自动停止)
  4. [诊断推理] 综合分析 → 结构化诊断报告
     主要诊断: 紧张型头痛 (ICD-11: 8A80)
     鉴别: 偏头痛 (8A60), 鼻窦炎 (CA01)
     置信度: medium | 严重程度: mild
  5. [治疗计划] 个性化治疗建议 + 随访安排
```

---

## 核心流程

```
用户输入主诉
    │
    ▼
┌──────────────┐
│  安全分诊     │  规则匹配 + LLM 双重检查
│              │  胸痛/卒中/大出血/过敏 → 立即告警
└──────┬───────┘
       ▼
┌──────────────┐
│  意图分类     │  diagnosis / research / planning
│  (LLM)       │  自动路由到正确的处理管道
└──────┬───────┘
       │
       ├─ research ──→ 医学知识问答 → 结束
       ├─ planning ──→ 治疗计划生成 → 结束
       │
       └─ diagnosis
            │
            ▼
       ┌────────────────────────────────┐
       │  动态问诊循环                   │
       │                                │
       │  LLM 生成问题 (按病史采集标准)    │
       │       ↓                        │
       │  interrupt() 等待用户回答       │
       │       ↓                        │
       │  LLM 提取关键信息 + 更新鉴别诊断  │
       │       ↓                        │
       │  信息充分? ─ No ─→ 继续循环      │
       │       │                        │
       │      Yes                       │
       └───────┬────────────────────────┘
               ▼
       ┌──────────────┐
       │  诊断推理     │  LLM 结构化报告 (ICD-11 + 证据)
       └──────┬───────┘
              ▼
       ┌──────────────┐
       │  治疗计划     │  药物讨论 + 非药物治疗 + 随访
       └──────────────┘
```

---

## 快速开始

### 环境

- Python >= 3.12
- 一个 LLM API Key（任选一个）：
  - [DeepSeek](https://platform.deepseek.com/)（推荐，性价比高）
  - [OpenAI](https://platform.openai.com/)
  - [智谱 GLM](https://open.bigmodel.cn/)
  - [Moonshot/Kimi](https://platform.moonshot.cn/)
  - [阿里百炼/Qwen](https://dashscope.aliyun.com/)

### 安装

```bash
git clone https://github.com/<your-name>/ClinicalMind.git
cd ClinicalMind
pip install -r requirements-console.txt
```

### 配置

```bash
cp .env.example .env
```

编辑 `.env`，填 **一个** LLM API Key：

```ini
# 以 DeepSeek 为例
DEEPSEEK_API_KEY=sk-your-key-here
DEFAULT_LLM_MODEL=deepseek-chat
```

### 运行

```bash
python console.py
```

```
============================================================
  ClinicalMind - LangGraph Console Edition
============================================================

Enter chief complaint / medical question:
> 我头痛一周了，太阳穴胀痛，有时候恶心想吐

[Round 1] (0 items collected)
  Current hypotheses:
    [++] 紧张型头痛
    [+-] 偏头痛

  Q1: 您头痛是突然开始还是慢慢加重的？
         [1] 突然开始
         [2] 慢慢加重
  Choice: 2

  ...

  ============ DIAGNOSIS REPORT ============
  Primary:   紧张型头痛
  Confidence: medium | Severity: mild
  ...
```

---

## 技术架构

使用 **[LangGraph](https://github.com/langchain-ai/langgraph)** 构建整个 Agent 工作流：

```
                   START
                     │
                ┌─────────┐
                │  entry  │  安全分诊 + 意图分类
                └────┬────┘
                     │
           route_after_entry
           ┌────────┼────────┐
           │        │        │
      interview  research  planning
      _loop      _node     _node
           │        │        │
           │        └── END ─┘
           │
      ┌─────────┐
      │diagnose │  结构化诊断
      └────┬────┘
           │
      ┌──────────────┐
      │treatment_plan│  治疗计划
      └──────┬───────┘
             │
            END
```

| 组件 | 技术 | 说明 |
|------|------|------|
| Agent 编排 | **LangGraph StateGraph** | 声明式图结构，节点 + 条件边 |
| 人机交互 | **interrupt()** | 暂停执行等待输入，`Command(resume=...)` 恢复 |
| LLM | **LangChain ChatOpenAI** | 兼容 OpenAI / DeepSeek / GLM / Moonshot / 百炼 |
| 状态管理 | **MemorySaver** | 自动 checkpoint，可升级 SqliteSaver 持久化 |
| 输出约束 | **Pydantic Schema** | 诊断报告/治疗计划通过 JSON Schema 校验 |

> 详细架构文档 → [LangGraph.md](./LangGraph.md)

---

## 项目结构

```
ClinicalMind/
├── clinicalmind_lg/          # LangGraph 工作流
│   ├── state.py              #   ClinicalState 定义 + reducer
│   ├── prompts.py            #   7 组中文 System Prompt
│   ├── llm.py                #   多 Provider LLM 客户端
│   ├── nodes.py              #   6 个 Agent 节点 + 路由函数
│   ├── graph.py              #   StateGraph 构建 + 编译
│   └── console.py            #   交互式控制台
│
├── console.py                # 入口
├── requirements-console.txt  # 依赖
├── .env.example              # 环境变量模板
├── LangGraph.md              # 工作流详细文档
├── LICENSE                   # MIT
└── README.md
```

---

## 功能特点

### Agent 智能体

| Agent | 职责 |
|-------|------|
| **MasterAgent** | 意图识别，自动路由（diagnosis/research/planning）|
| **DiagnosisAgent** | 症状分析 + 结构化诊断报告（ICD-11 编码）|
| **PlanningAgent** | 个性化治疗计划（药物讨论/康复/随访）|
| **ResearchAgent** | 医学知识查询 |
| **MonitoringAgent** | 患者随访跟踪 |

### 安全机制

- **双层分诊** — 规则关键词毫秒级匹配 + LLM 深度分析
- **急诊识别** — 胸痛+呼吸困难、卒中、大出血、严重过敏、妊娠危险信号
- **危险信号追踪** — 问诊过程中持续监控

### 问诊引擎

- **动态自适应** — 不预设脚本，根据已收集信息智能生成下一步
- **鉴别诊断驱动** — 每轮更新诊断假设，引导信息收集方向
- **反冗余** — 语义去重 + 指纹去重
- **中国执业医师标准** — 现病史、既往史、个人史、家族史、用药史

---

## 文档

| 文档 | 内容 |
|------|------|
| [LangGraph.md](./LangGraph.md) | 工作流详解：State 定义、6 个节点、路由、interrupt 机制、扩展指南 |

---

## License

MIT License — 详见 [LICENSE](./LICENSE)。
