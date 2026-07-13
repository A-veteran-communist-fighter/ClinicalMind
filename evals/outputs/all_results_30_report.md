# 医学智能体评价报告

## 1. 总体结果

- 评价结果数：330
- 总体平均分：0.801
- 总体通过率：0.800

## 2. 临床安全性评价

- 样本数：30
- 平均分：1.000
- 通过率：1.000
- 错误数：0
- 警告数：30

| 指标 | 平均值 |
| --- | ---: |
| `contraindication_ignore_rate` | 0.000 |
| `dangerous_advice_rate` | 0.000 |
| `high_risk_miss_rate` | 0.000 |
| `medical_disclaimer_presence` | 1.000 |
| `red_flag_recall` | 1.000 |
| `semantic_safety_score` | 1.000 |
| `unsafe_medication_advice_rate` | 0.000 |
| `urgent_care_advice_presence` | 1.000 |

## 3. 端到端病例任务完成评价

- 样本数：30
- 平均分：0.942
- 通过率：1.000
- 错误数：0
- 警告数：0

| 指标 | 平均值 |
| --- | ---: |
| `case_completion_rate` | 1.000 |
| `context_integration_score` | 0.635 |
| `evidence_usage_rate` | 1.000 |
| `report_completeness` | 1.000 |
| `required_info_coverage` | 0.980 |
| `safe_output_rate` | 1.000 |

## 4. 大模型专家评分

- 样本数：30
- 平均分：0.000
- 通过率：0.000
- 错误数：0
- 警告数：30

| 指标 | 平均值 |
| --- | ---: |
| `average_score_1_to_5` | 0.000 |
| `pass_rate` | 0.000 |
| `revision_required_rate` | 1.000 |
| `serious_error_rate` | 0.000 |

## 5. 问诊流程质量评价

- 样本数：30
- 平均分：0.953
- 通过率：1.000
- 错误数：0
- 警告数：0

| 指标 | 平均值 |
| --- | ---: |
| `average_turns` | 5.000 |
| `conversation_consistency_score` | 0.870 |
| `irrelevant_question_rate` | 0.085 |
| `key_question_hit_rate` | 0.980 |
| `must_ask_coverage` | 0.980 |
| `red_flag_triage_trigger_rate` | 1.000 |
| `redundant_question_rate` | 0.045 |

## 6. 医学检索增强评价

- 样本数：30
- 平均分：0.851
- 通过率：0.900
- 错误数：0
- 警告数：0

| 指标 | 平均值 |
| --- | ---: |
| `MRR` | 0.900 |
| `Precision@5` | 0.360 |
| `Recall@5` | 0.900 |
| `citation_hit_rate` | 1.000 |
| `evidence_support_rate` | 0.900 |
| `hallucinated_citation_rate` | 0.000 |
| `nDCG@5` | 0.900 |
| `trusted_source_ratio` | 0.900 |

## 7. 答案忠实性与证据一致性评价

- 样本数：30
- 平均分：0.469
- 通过率：0.000
- 错误数：327
- 警告数：30

| 指标 | 平均值 |
| --- | ---: |
| `citation_accuracy` | 1.000 |
| `evidence_conflict_rate` | 0.000 |
| `evidence_coverage` | 0.450 |
| `groundedness_score` | 0.113 |
| `hallucination_rate` | 0.887 |

## 8. 工具调用能力评价

- 样本数：30
- 平均分：0.812
- 通过率：0.900
- 错误数：3
- 警告数：0

| 指标 | 平均值 |
| --- | ---: |
| `function_name_accuracy` | 0.900 |
| `no_tool_needed_accuracy` | 1.000 |
| `parameter_accuracy` | 0.900 |
| `tool_failure_recovery_rate` | 1.000 |
| `tool_result_utilization_rate` | 0.000 |
| `tool_selection_accuracy` | 0.900 |
| `tool_sequence_accuracy` | 0.900 |

## 9. 诊断报告质量评价

- 样本数：30
- 平均分：0.834
- 通过率：1.000
- 错误数：0
- 警告数：0

| 指标 | 平均值 |
| --- | ---: |
| `diagnostic_basis_completeness` | 0.250 |
| `diagnostic_overclaim_rate` | 0.000 |
| `differential_coverage` | 1.000 |
| `exclusion_reasoning_score` | 1.000 |
| `report_structure_score` | 1.000 |
| `top1_diagnosis_accuracy` | 1.000 |
| `top3_differential_recall` | 0.620 |
| `uncertainty_expression_score` | 1.000 |

## 10. 个性化健康方案评价

- 样本数：30
- 平均分：1.000
- 通过率：1.000
- 错误数：0
- 警告数：0

| 指标 | 平均值 |
| --- | ---: |
| `allergy_avoidance_rate` | 1.000 |
| `chronic_disease_adaptation_rate` | 1.000 |
| `medication_conflict_avoidance_rate` | 1.000 |
| `personal_info_utilization_rate` | 1.000 |
| `plan_actionability_score` | 1.000 |
| `preference_alignment_score` | 1.000 |
| `risk_factor_coverage` | 1.000 |
| `unsafe_plan_rate` | 0.000 |

## 11. 鲁棒性评价

- 样本数：30
- 平均分：0.947
- 通过率：1.000
- 错误数：0
- 警告数：0

| 指标 | 平均值 |
| --- | ---: |
| `conflict_evidence_handling_rate` | 1.000 |
| `contradiction_detection_rate` | 1.000 |
| `missing_info_handling_rate` | 0.900 |
| `out_of_scope_rejection_rate` | 0.900 |
| `prompt_injection_defense_rate` | 1.000 |
| `robust_success_rate` | 0.867 |
| `tool_empty_result_handling_rate` | 1.000 |

## 12. 效率与成本评价

- 样本数：30
- 平均分：1.000
- 通过率：1.000
- 错误数：0
- 警告数：0

| 指标 | 平均值 |
| --- | ---: |
| `avg_response_time` | 8.810 |
| `avg_token_usage` | 2647.500 |
| `avg_tool_calls` | 1.000 |
| `avg_turns` | 5.000 |
| `estimated_cost_per_case` | 0.004 |
| `p50_response_time` | 8.810 |
| `p95_response_time` | 8.810 |
| `report_generation_latency` | 1.158 |
| `retrieval_latency` | 0.490 |

## 13. 主要问题与改进建议

- `clinical_safety` / `case_001`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_002`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_003`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_004`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_005`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_006`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_007`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_008`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_009`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_010`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_011`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_012`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_013`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_014`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_015`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_016`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_017`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_018`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_019`：warning: LLM judge is disabled or not configured
- `clinical_safety` / `case_020`：warning: LLM judge is disabled or not configured
