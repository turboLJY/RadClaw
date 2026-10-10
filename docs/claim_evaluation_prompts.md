# RadClaw：Claim 抽取、幻觉判定、匹配召回与证据关系 Prompt

整理日期：2026-09-30。本文依据当前项目源码整理，供论文方法描述和附录使用。代码块保留实际 prompt 的文字、输出字段和动态占位符；Python `.format()` 模板中用于转义的双花括号，展示为模型实际接收的单花括号。说明文字不是 prompt 的组成部分。

## 功能与实现对应

| 功能 | 当前实现 | Prompt / 入口 |
| --- | --- | --- |
| Claim 抽取 | LLM 分别抽取生成报告与参考报告的 Findings、Impression | `CLAIM_EXTRACTION_PROMPT` |
| Claim 幻觉判断 | 支持判定、规则匹配、内部一致性和建议支持检查后，由 LLM 进行 H1–H10 分类，并进行程序后处理 | `TAXONOMY_CLASSIFICATION_PROMPT`，以及两个辅助模板 |
| Generated claim 与 reference claim 匹配、遗漏统计 | Findings 使用同义词归一化、相似度评分和贪心一对一匹配；最终分类输出也包含匹配和遗漏字段 | `match_findings_crimson()`；没有独立的匹配召回 prompt |
| Claim–evidence relation 判断 | 对每个生成 claim，结合参考报告、参考 claims 和可用证据给出五类支持关系 | `SUPPORT_VERIFICATION_PROMPT` |

主要来源：[hallucination_detection/detect.py](../hallucination_detection/detect.py)。实际流程入口是 `evaluate_case()`（第 2841 行）。执行顺序为：claim 抽取 → Stage A 支持判定 → Stage B Findings 匹配 → Stage C 内部一致性与建议支持检查 → Stage D 幻觉分类及后处理。

## 0. 主评估流程共用的 System Prompt

来源：`hallucination_detection/detect.py`，`OpenAIBackbone.call_json()`（第 1790 行附近）。下文第 1、2、4 节的主评估模板均作为 **user message** 发送，并共用以下 system message。附录中的独立标注脚本使用各自的 system message。

```text
You are a careful radiology evaluation system. Return valid JSON only.
```

启用图像输入时，user message 包含模板文本、逐张图像的说明文本和图像内容块。图像说明的文本格式为：

```text
Image {index}: image_id={image_id}; path={rel_path}; description={description_or_default}
```

`index` 从 1 开始；描述为空时，`description_or_default` 为 `No description provided.`。Stage A 在 `include_images=True` 且 `image_stage="all"` 时附图；最终 taxonomy 在启用图像且 `image_stage` 为 `taxonomy` 或 `all` 时附图。抽取、内部一致性和建议支持检查不直接附图。

## 1. Claim 抽取

来源：`hallucination_detection/detect.py` 第 1545 行，`CLAIM_EXTRACTION_PROMPT`。

调用：`llm_extract_section_claims()`（第 1980 行）及 `llm_extract_all_claims()`（第 2015 行）。生成报告和参考报告使用同一个模板，分别处理各自非空的 Findings 与 Impression。

### User Prompt 原文

```text
You are an expert radiology information extraction system.

Extract atomic clinical claims from the following radiology report section.

Rules:
1. Each claim should contain one clinical assertion only.
2. Split compound sentences into separate claims.
3. Preserve negation, uncertainty, laterality, location, severity, measurement, and comparison.
4. Classify each claim into one of:
   positive_finding, negative_finding, attribute_claim, measurement_claim,
   comparison_claim, impression_summary, diagnostic_inference,
   etiology_claim, certainty_claim, recommendation_claim, normality_claim.
5. Return valid JSON only.

Input section:
Section name: {section_name}
Text:
{section_text}

Return this JSON object only:
{
  "claims": [
    {
      "raw_text": "...",
      "claim_type": "...",
      "core_finding": "...",
      "polarity": "positive | negative | uncertain",
      "anatomy": "...",
      "laterality": "left | right | bilateral | midline | none | unknown",
      "location_detail": "...",
      "severity": "...",
      "measurement": "...",
      "descriptor": "...",
      "temporal": "...",
      "certainty": "...",
      "recommendation": "..."
    }
  ]
}
```

### 输入与输出说明

| 变量 | 运行时内容 |
| --- | --- |
| `{section_name}` | `Findings` 或 `Impression` |
| `{section_text}` | 对应报告段落文本，经代码规范化后填入 |

Claim ID 由程序补充：生成 Findings 为 `G_F_001` 等，生成 Impression 为 `G_I_001` 等，参考 Findings 为 `R_F_001` 等，参考 Impression 为 `R_I_001` 等。模型输出的 `claims` 还会经过字段规范化。

## 2. Claim 幻觉判断

### 2.1 最终 H1–H10 分类 User Prompt

来源：`hallucination_detection/detect.py` 第 1653 行，`TAXONOMY_CLASSIFICATION_PROMPT`；调用函数为 `llm_classify_taxonomy()`（第 2791 行）。

```text
You are an expert radiology report hallucination evaluator.

Classify claim-level medical hallucinations/errors for one radiology report generation case.
Use the H1-H10 taxonomy below and the provided LLM stage outputs.

Hallucination taxonomy:
- H1_fabricated_finding: generated abnormality/finding is not supported by reference or evidence.
- H2_omitted_finding: clinically relevant positive reference finding is not mentioned by generated report. H2 is an omission event, not a generated-claim label.
- H3_negation_error: generated and reference claims share the same finding/anatomy but opposite polarity.
- H4_location_or_laterality_error: correct finding but wrong organ, side, lobe, or clinically meaningful location.
- H5_attribute_error: correct finding but wrong size, number, severity, extent, morphology, density, signal, margin, distribution, or measurement.
- H6_comparison_error: wrong temporal comparison such as new/stable/increased/decreased/resolved.
- H7_unsupported_diagnostic_inference: diagnosis, etiology, causal claim, or overinterpretation is not supported by findings/evidence.
- H8_findings_impression_inconsistency: generated Impression contradicts or is not supported by generated Findings.
- H9_certainty_calibration_error: certainty is too strong or too weak relative to reference/evidence.
- H10_unsupported_recommendation: recommendation is unsupported by findings, patient context, guidelines, or accepted practice.

Primary type priority:
1. H3_negation_error
2. H8_findings_impression_inconsistency
3. H1_fabricated_finding
4. H2_omitted_finding
5. H4_location_or_laterality_error
6. H6_comparison_error
7. H5_attribute_error
8. H7_unsupported_diagnostic_inference
9. H9_certainty_calibration_error
10. H10_unsupported_recommendation

Rules:
- Return one claim_level_results item for every generated claim id, exactly once.
- Put omissions only in omission_events, not as H2 labels on generated claims.
- Do not double count a laterality/location mismatch as H1 plus H2; use H4.
- Do not double count overinterpretation of a matched finding as H1; use H7.
- Use Stage A support status to decide whether a generated claim is supported, but assign H-types using the taxonomy rules.
- Stage B CRIMSON-style findings results are deterministic pre-classification results. Preserve their generated-claim H-types and H2 omission events unless the relevant generated claim is unverifiable.
- Image evidence contains image_id, description, and attached image blocks. Use both the image pixels and the image descriptions when available.
- Use internal consistency results for H8.
- Use recommendation support results for H10.
- Use clinical severity labels only from: critical, major, minor, negligible. Use null if uncertain.
- If support_status is unverifiable and there is no contradiction or clear taxonomy error, do not count it as an error.

Case input JSON:
{case_payload}

Return valid JSON only with this schema:
{
  "claim_level_results": [
    {
      "claim_id": "G_F_001",
      "section": "Findings | Impression",
      "claim_text": "...",
      "claim_type": "...",
      "support_status": "entailed | partially_entailed | not_entailed | contradicted | unverifiable",
      "is_error": true,
      "primary_hallucination_type": "H1_fabricated_finding | ... | null",
      "hallucination_types": ["H1_fabricated_finding"],
      "matched_reference_claims": ["R_F_001"],
      "evidence_used": ["..."],
      "clinical_severity": "critical | major | minor | negligible | null",
      "brief_reason": "...",
      "error_details": {}
    }
  ],
  "omission_events": [
    {
      "reference_claim_id": "R_F_001",
      "reference_claim_text": "...",
      "hallucination_type": "H2_omitted_finding",
      "matched_generated_claims": ["G_F_001"],
      "clinical_severity": "critical | major | minor | negligible | null",
      "brief_reason": "..."
    }
  ]
}
```

`{case_payload}` 是 JSON，包含 `case_id`、`generated_report`、`reference_report`、`generated_claims`、`reference_claims`、`stage_a_support_results`、`internal_consistency_results`、`recommendation_support_results`、`stage_b_findings_results`、`stage_b_omission_events`、`evidence` 和 `config`。其中 `config` 包含 `strict_negative_claims` 与 `include_negative_omissions`。

### 2.2 Findings–Impression 内部一致性 User Prompt

来源：`hallucination_detection/detect.py` 第 1611 行，`INTERNAL_CONSISTENCY_PROMPT`；调用函数为 `llm_internal_consistency()`（第 2138 行）。用于辅助 H8 判断。

```text
You are an expert radiology report evaluator.

Determine whether the Impression claim is supported by the generated Findings.

Generated Findings claims:
{findings_claims}

Generated Impression claim:
{impression_claim}

Return valid JSON only:
{
  "support_status": "entailed | partially_entailed | not_entailed | contradicted | unverifiable",
  "contradicting_finding_claims": ["..."],
  "brief_reason": "..."
}
```

`{findings_claims}` 为所有生成 Findings claims 的 JSON；`{impression_claim}` 为当前生成 Impression claim 的 JSON。每个生成 Impression claim 单独调用一次。

### 2.3 Recommendation 支持判断 User Prompt

来源：`hallucination_detection/detect.py` 第 1629 行，`RECOMMENDATION_SUPPORT_PROMPT`；调用函数为 `llm_recommendation_support()`（第 2165 行）。用于辅助 H10 判断。

```text
You are an expert radiology report evaluator.

Determine whether the generated recommendation is clinically supported.

Recommendation claim:
{recommendation_claim}

Available findings:
{findings_claims}

Patient context:
{patient_context}

Guideline or medical evidence:
{medical_evidence}

Return valid JSON only:
{
  "recommendation_status": "supported | partially_supported | unsupported | contradicted | unverifiable",
  "supporting_evidence": ["..."],
  "brief_reason": "..."
}
```

`{recommendation_claim}` 为当前生成的 `recommendation_claim`；`{findings_claims}` 合并生成报告和参考报告的 Findings claims；`{patient_context}` 为患者上下文；`{medical_evidence}` 包含 `retrieved_medical_evidence`、`similar_case_evidence` 和 `image_evidence`。这些变量均通过 JSON 序列化填入。

### 2.4 论文描述需要保留的后处理规则

来源：`sanitize_claim_results()`（第 2578 行）、`apply_unverifiable_suppression()`（第 2561 行）和 `compute_case_metrics()`（第 2724 行）。

- 最终结果会合并 Stage B 的幻觉标签和匹配 reference ID，并补入 Stage B 遗漏事件。
- 内部一致性结果为 `not_entailed` 或 `contradicted` 时补充 H8；建议支持结果为 `unsupported` 或 `contradicted` 时补充 H10。
- 最终 `support_status` 为 `unverifiable` 时，代码会清空该生成 claim 的幻觉标签，将 `is_error` 设为 `False`。这是程序规则，比 prompt 中带条件的表述更强。
- `hallucination_rate` / `generated_claim_error_rate` 的分母为全部 generated claims；`unverifiable` 不进入错误数，但仍包含在该分母中，并另行报告 `unverifiable_rate`。
- H2 作为 reference 侧的遗漏事件保存，不作为 generated claim 的幻觉标签。

## 3. Generated claim 与 reference claim 匹配、召回／遗漏

### 3.1 当前代码没有独立的匹配召回 Prompt

当前 `evaluate_case()` 调用 `run_stage_b_findings()`，后者使用 `match_findings_crimson()` 进行确定性匹配。最终 taxonomy prompt 中虽要求输出 `matched_reference_claims` 与 `omission_events`，但没有另一个专门计算 claim recall 的 LLM prompt。

来源：`hallucination_detection/detect.py` 中 `crimson_match_score()`（第 2280 行）、`crimson_candidate_match()`（第 2304 行）、`match_findings_crimson()`（第 2312 行）及 `run_stage_b_findings()`（第 2436 行）。

### 3.2 实际匹配规则

1. 只对 Stage B 接收的 Findings claims 配对。类型包括 `positive_finding`、`negative_finding`、`attribute_claim`、`measurement_claim`、`comparison_claim`、`normality_claim`、`diagnostic_inference`、`etiology_claim` 和 `certainty_claim`。
2. 使用源码中的放射学发现与解剖同义词表，规范化核心发现和解剖位置。
3. 对每个 generated–reference 候选对计算分数：核心发现完全相同加 0.72；否则核心发现 token overlap ≥ 0.75 加 0.55，≥ 0.5 加 0.42；原文 token overlap 贡献 `min(overlap × 0.25, 0.18)`；解剖兼容加 0.12；解剖键非空且相同再加 0.08。最终分数上限为 1.0。
4. 核心发现相同直接成为候选；否则要求分数不低于 `max(0.35, min(config.min_match_score, 0.5))`。`min_match_score` 默认 0.45，因此默认候选阈值为 0.45。
5. 候选按分数降序排序，贪心选择一对一配对；每个 generated claim 和 reference claim 最多使用一次。
6. 对成功配对的 claims 再检查否定、位置、属性、比较、过度推断和确定性等错误。因此，“匹配成功”本身不等于“完全支持”或“无幻觉”。

### 3.3 遗漏与召回的统计口径

未匹配且符合 `clinically_relevant_reference_claim()` 条件的 reference Findings claims 形成 Stage B H2 遗漏事件。默认条件涵盖非 negative 的 `positive_finding`、`diagnostic_inference`、`etiology_claim`、`comparison_claim`、`measurement_claim` 和 `certainty_claim`；启用 `include_negative_omissions` 时还纳入 `negative_finding` 和 `normality_claim`。

最终分类还可返回遗漏事件，并与 Stage B 遗漏结果合并。现有 `compute_case_metrics()` 输出：

```text
omission_rate = num_omission_events / num_reference_claims
```

这里的 `num_reference_claims` 是全部参考 claims 数，包括 Findings 和 Impression。当前函数没有直接输出 `claim_recall` 字段。若论文另定义匹配召回率，需要明确参考 claim 集合、是否要求匹配后语义正确、是否纳入阴性发现和 Impression；不能直接把代码中的 `1 - omission_rate` 描述为已实现的“临床相关阳性 Findings 召回率”。

## 4. Claim–Evidence Relation 判断

来源：`hallucination_detection/detect.py` 第 1586 行，`SUPPORT_VERIFICATION_PROMPT`；调用函数为 `llm_verify_support()`（第 2104 行）。每个生成 claim 单独调用一次。

### User Prompt 原文

```text
You are an expert radiology report evaluator.

Your task is to determine whether a generated clinical claim is supported by the provided reference/evidence.

Generated claim:
{claim}

Reference/evidence:
{premise}

Definitions:
- entailed: the evidence fully supports the claim.
- partially_entailed: the evidence supports part of the claim but not all details, certainty, location, measurement, or interpretation.
- not_entailed: the evidence does not support the claim.
- contradicted: the evidence explicitly contradicts the claim.
- unverifiable: the evidence is insufficient to judge.

Return valid JSON only:
{
  "support_status": "entailed | partially_entailed | not_entailed | contradicted | unverifiable",
  "supporting_evidence": ["..."],
  "brief_reason": "..."
}
```

### 关系标签

| 源码标签 | 中文含义 | Prompt 中的判定依据 |
| --- | --- | --- |
| `entailed` | 完全支持 | 证据完全支持该 claim |
| `partially_entailed` | 部分支持 | 只支持部分内容，细节、确定性、位置、测量或解释未被完全支持 |
| `not_entailed` | 不支持 | 证据不支持该 claim |
| `contradicted` | 矛盾 | 证据明确反驳该 claim |
| `unverifiable` | 无法核实 | 证据不足以作出判断 |

### 动态输入

`{claim}` 由 `claim_prompt_dict()`（第 2067 行）序列化，字段为：`claim_id`、`section`、`raw_text`、`claim_type`、`core_finding`、`polarity`、`anatomy`、`laterality`、`location_detail`、`severity`、`measurement`、`descriptor`、`temporal`、`certainty`、`recommendation`。

`{premise}` 的字段及内容：

| 字段 | 内容 |
| --- | --- |
| `reference_claims` | 所有参考 claims 的结构化表示 |
| `reference_report` | 参考报告 Findings 与 Impression |
| `generated_report_context` | 生成报告 Findings 与 Impression，作为上下文 |
| `image_evidence` | 图像 ID、描述等图像证据信息 |
| `retrieved_medical_evidence` | 可用检索医学证据 |
| `similar_case_evidence` | 可用相似病例证据 |
| `prior_report` | 可用既往报告 |
| `patient_context` | 患者上下文 |
| `image_manifest` | 图像文件清单及元信息 |

这些证据由 `collect_optional_evidence()`（第 1886 行）汇集；医学检索证据和相似病例证据取自 `summary.json`、`result.json` 的对应顶层字段，患者上下文和既往报告也从这些文件获取；缺失字段使用空值。图像来自病例图像清单及其描述。

这一调用输出的是“单个 claim 相对于整组 reference/evidence”的总体关系，当前实现没有在此逐一为每个 claim–单条 evidence 配对输出关系标签。论文若采用“claim–evidence relation”这一术语，应说明这里的 evidence 是聚合证据。

## 附录 A. 中文 Ground-truth Claim 抽取模板

这是独立的中文标注数据处理脚本所用模板，与第 1 节主评估流程的英文模板分别记录。

来源：[scripts/extract_ground_truth_claims.py](../scripts/extract_ground_truth_claims.py)，`build_system_prompt()`（第 200 行）与 `build_user_prompt()`（第 209 行）。

### System Prompt

```text
你是严谨的放射科医学文本标注助手。你的任务是把中文影像报告拆解为原子医学 claim。原子 claim 是一个可以独立核查的医学结论，必须尽量保留原文中的解剖部位、侧别、程度、否定、可能性、鉴别诊断和因果/解释关系。不要加入原文没有的信息，不要翻译成英文，不要输出非医学寒暄或说明。
```

### User Prompt

```text
请将下面 JSON 中的 FINDINGS 与 IMPRESSION 文本分别拆解为原子医学 claim。
输出要求：
1. 只输出一个 JSON object，不要 Markdown。
2. JSON schema 必须是 {"claims": [{"section": "FINDINGS|IMPRESSION", "claim": "..."}]}。
3. section 必须严格使用 FINDINGS 或 IMPRESSION，表示 claim 来自哪个输入字段。
4. 一个 claim 只表达一个医学结论；并列结论、列表项和否定发现要拆开。
5. 修饰信息必须留在同一个 claim 中，例如部位、侧别、程度、否定、可疑/可能、鉴别诊断。
6. 如果同一结论同时出现在 FINDINGS 和 IMPRESSION 中，分别保留，不要跨 section 去重。
7. 如果某个 section 为空，不要为它生成 claim。

输入 JSON：
{
  "FINDINGS": "{findings_text}",
  "IMPRESSION": "{impression_text}"
}
```

末尾输入对象由 `json.dumps(payload, ensure_ascii=False, indent=2)` 构造；这里用 `{findings_text}` 和 `{impression_text}` 表示实际文本，运行时会正确进行 JSON 转义。

## 附录 B. 英文基线报告 Claim 抽取模板

来源：[scripts/process_qwen_baseline_reports_to_chinese_claims.py](../scripts/process_qwen_baseline_reports_to_chinese_claims.py)，`build_claim_extraction_system_prompt()`（第 383 行）与 `build_claim_extraction_user_prompt()`（第 395 行）。该脚本先抽取英文 claims，再执行中文翻译；以下是其中的抽取模板。

### System Prompt

```text
You are a meticulous radiology text annotation assistant. Your task is to split English chest X-ray report text into atomic medical claims. An atomic claim is one independently checkable medical assertion. Preserve anatomy, laterality, severity, negation, uncertainty, differential diagnoses, device/hardware details, and causal or explanatory relationships when they are present in the source. Do not add, infer, translate, summarize away, or normalize beyond what is needed to make a concise standalone English claim.
```

### User Prompt

```text
Split the FINDINGS and IMPRESSION text in the JSON below into atomic medical claims.
Output requirements:
1. Output exactly one JSON object and no Markdown.
2. The JSON schema must be {"claims": [{"section": "FINDINGS|IMPRESSION", "claim_text": "..."}]}.
3. section must be exactly FINDINGS or IMPRESSION and must reflect the source input field.
4. Keep claim_text in English.
5. Split conjoined observations and conjoined negated findings into separate claims when each can be checked independently.
6. Keep modifiers in the same claim, including anatomy, side, degree, negation, uncertainty, and differential diagnoses.
7. Exclude non-medical introduction, technique/view/acquisition statements, and patient-facing explanatory parentheticals unless they are required for the claim.
8. If a section is empty, do not create claims for that section.
9. If the same claim appears in both FINDINGS and IMPRESSION, keep both entries with their respective section values.

Input JSON:
{
  "FINDINGS": "{findings_text}",
  "IMPRESSION": "{impression_text}"
}
```

末尾输入对象同样由 `json.dumps(payload, ensure_ascii=False, indent=2)` 构造，两个文本占位符仅用于展示输入结构。
