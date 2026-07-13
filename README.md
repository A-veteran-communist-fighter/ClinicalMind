# ClinicalMind

> 基于 LangGraph 的多智能体医学问诊系统

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/framework-LangGraph-green.svg)](https://github.com/langchain-ai/langgraph)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](./LICENSE)

---

## 项目背景与动机

2025 年以来，大语言模型在医疗领域的应用呈爆发式增长。但在实际调研中发现了几个普遍问题：

1. **大多数"AI 医生"只是单轮问答**——真正的临床问诊是多轮、动态、循证的信息收集过程
2. **工程复杂度淹没了核心逻辑**——开源医疗 AI 项目几乎都是 FastAPI + PostgreSQL + Redis + React 全家桶，Agent 核心逻辑藏在服务层下面
3. **缺乏结构化临床思维**——没有按照临床标准（如中国执业医师病史采集规范）来设计问诊流程
4. **安全分诊被忽视**——胸痛合并呼吸困难的患者应该立刻去急诊，但大多数 AI 医疗项目缺乏这一层

ClinicalMind 试图用一个极简但完整的实现回答这些问题：把临床问诊的标准化流程和 LLM 的自然语言理解结合起来，用 LangGraph 状态图编排，让 Agent 在正确的框架内做正确的事。

**⚠️ 医疗免责声明：本系统输出仅供参考，不能替代专业医疗诊断。如有胸痛、呼吸困难、意识障碍等急危重症，请立即就医。**

---

## 它做什么？

```
用户: "我最近一周持续头痛，额头和太阳穴胀痛，有时恶心，睡眠不好"
  + lab C:\reports\blood_test.jpg    （可选：上传化验单图片）

系统:
  1. [多模态解析] 视觉模型识别化验单 → 提取异常指标（空腹血糖8.2↑、肌酐↑）
  2. [安全分诊] 规则 + LLM 双重检查 — 结合化验结果判断 — 非急诊，继续
  3. [意图分类] LLM 识别意图 — diagnosis
  4. [动态问诊] 按中国医师病史采集标准逐轮追问
     Q1: "头痛是突然开始还是慢慢加重的？"
     Q2: "头痛是持续性胀痛还是阵发性跳痛？"
     ... (信息充分后自动停止)
  5. [诊断推理] 结构化报告（化验数据已纳入分析）
     主要诊断: 紧张型头痛 (ICD-11: 8A80)
     鉴别: 偏头痛 (8A60), 鼻窦炎 (CA01)
  6. [治疗计划] 个性化建议 + 随访安排
```

---

## 快速开始

**只需 Python 3.12+ + 一个 LLM API Key。**

```bash
git clone https://github.com/A-veteran-communist-fighter/ClinicalMind.git
cd ClinicalMind
pip install -r requirements-console.txt
cp .env.example .env
# 编辑 .env，填一个 LLM API Key
python console.py
```

支持的 LLM Provider（任选一个）：

| Provider | 环境变量 | 推荐理由 |
|----------|---------|---------|
| DeepSeek | `DEEPSEEK_API_KEY` | 性价比最高，推荐 |
| OpenAI | `OPENAI_API_KEY` | 生态最成熟 |
| 智谱 GLM | `GLM_API_KEY` | 国内合规 |
| Moonshot/Kimi | `MOONSHOT_API_KEY` | 中文能力强 |
| 阿里百炼 | `DASHSCOPE_API_KEY` | 阿里云生态 |

### 多模态化验单识别（可选）

如果需要上传化验单/检查报告图片，需额外配置视觉模型（独立于主 LLM）：

```ini
# .env 中添加
VISION_API_KEY=sk-your-openai-key     # 视觉模型 API Key（需支持视觉的模型）
VISION_BASE_URL=https://api.openai.com/v1
VISION_MODEL=gpt-4o                   # 默认 gpt-4o
```

未配置时 `lab` 命令自动降级不启用，不影响其他功能。

---

## 核心流程

```
用户输入主诉
    │
    ▼
┌──────────┐
│  安全分诊 │  规则正则 + LLM 双重检查
│          │  胸痛/卒中/大出血/过敏 → 立即告警 → 跳过问诊 → 急诊报告
└────┬─────┘
     ▼
┌──────────┐
│  意图分类 │  diagnosis / research / planning
│  (LLM)   │  自动路由
└────┬─────┘
     │
     ├─ research ──→ 医学问答 → 结束
     ├─ planning ──→ 治疗计划 → 结束
     │
     └─ diagnosis
          │
          ▼
     ┌────────────────────────┐
     │  动态问诊循环            │
     │  LLM 生成问题 → 展示    │
     │  ← 用户回答             │
     │  LLM 提取关键信息        │
     │  更新鉴别诊断 ←──┐       │
     │  判断是否充分 ──Yes─→    │
     └────────────────────────┘
          │
          ▼
     ┌──────────┐
     │  诊断推理  │  LLM 结构化报告 (ICD-11)
     └────┬─────┘
          ▼
     ┌──────────┐
     │  治疗计划  │  药物讨论 + 非药物治疗 + 随访
     └──────────┘
```

---

## 技术架构

使用 **LangGraph** 声明式状态图构建整个 Agent 工作流：

```
START → [entry] → route_after_entry
                     ├─ diagnosis → [interview_generate] ←──┐
                     │                → END (暂停)          │
                     │                ← 用户回答            │
                     │              [interview_process] ────┘
                     │                → [diagnose] → [treatment_plan] → END
                     ├─ research → [research] → END
                     └─ planning → [standalone_planning] → END
```

人机交互通过**状态驱动**实现——生成问题后 graph 在 END 节点暂停，console 读取 `state["current_questions"]` 展示给用户，收集答案后填入 `human_response` 重新运行 graph。

| 组件 | 技术 | 说明 |
|------|------|------|
| Agent 编排 | LangGraph StateGraph | 6 个节点 + 3 个条件路由 |
| LLM 客户端 | LangChain ChatOpenAI | 兼容 5 个 Provider |
| 状态管理 | MemorySaver | 自动 checkpoint |
| 人机交互 | 状态驱动 | 生成问题 → 暂停 → 回答 → 继续 |
| 安全分诊 | 规则正则 + LLM 双层 | 毫秒级关键词 + 深度分析 |

详细架构文档 → [LangGraph.md](./LangGraph.md)

---

## 项目结构

```
ClinicalMind/
├── clinicalmind_lg/          # LangGraph 工作流
│   ├── state.py              #   ClinicalState 定义 (23字段，含 lab_reports)
│   ├── prompts.py            #   8 组中文 System Prompt（含化验单解析）
│   ├── llm.py                #   多 Provider LLM + 视觉模型（独立配置，降级可用）
│   ├── nodes.py              #   7 个节点 + 化验单解析 + 3 个路由函数
│   ├── graph.py              #   StateGraph 构建 + MemorySaver
│   └── console.py            #   交互式控制台
│
├── console.py                # 入口: python console.py
├── requirements-console.txt  # 依赖清单
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
| MasterAgent | 意图识别，自动路由 |
| DiagnosisAgent | 症状分析 + 结构化诊断 (ICD-11) |
| PlanningAgent | 个性化治疗计划 |
| ResearchAgent | 医学知识查询 |
| MonitoringAgent | 患者随访跟踪 |

### 安全机制

- **双层分诊**：规则正则毫秒级初筛 + LLM 深度分析
- **急诊识别**：胸痛+呼吸困难、卒中、大出血、严重过敏、妊娠危险信号
- **危险信号追踪**：问诊过程中持续监控

### 化验单识别（多模态）

- **视觉模型独立配置**：不与主 LLM 绑定，未配置时自动降级
- 支持常见图片格式（PNG/JPEG/WEBP/GIF），自动检测 MIME 类型
- LLM 提取结构化指标：名称、数值、单位、参考范围、异常分级、临床意义
- 问诊中实时上传：`lab <image_path>` 随时注入化验数据
- 解析结果自动纳入诊断推理，危急值标记为急诊级危险信号

### 问诊引擎

- 不预设脚本，根据已收集信息 LLM 动态生成下一步问题
- 每轮更新鉴别诊断假设，引导信息收集方向
- 语义去重 + 指纹去重，不重复问同一个问题
- 最大轮次保护，避免无限循环

---

## License

MIT License — 详见 [LICENSE](./LICENSE)。
