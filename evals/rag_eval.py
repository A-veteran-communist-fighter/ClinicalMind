#!/usr/bin/env python3
"""ClinicalMind RAG Evaluation — Faithfulness, Answer Relevance, Context Relevance.

Uses LLM-as-judge to score each RAG metric on a medical QA test dataset.
No external dependency beyond openai. Each metric is scored 0-1 by the LLM
with a structured rubric, then verified for consistency.

Metrics:
  - Faithfulness:     Is every claim in the answer supported by the context? (0-1)
  - Answer Relevance: Does the answer directly address the question? (0-1)
  - Context Relevance: Are the retrieved docs relevant to the question? (0-1)

Usage:
    # Ensure .env has a valid LLM API key
    python -m evals.rag_eval
"""

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Load .env ────────────────────────────────────────────────────────────
_env_path = Path(__file__).parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key not in os.environ:
                os.environ[key] = value

from openai import AsyncOpenAI


# ═══════════════════════════════════════════════════════════════════════════
# LLM Client
# ═══════════════════════════════════════════════════════════════════════════

def _get_client() -> AsyncOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com/v1"
    if not api_key:
        raise RuntimeError("No LLM API key found. Set one in .env")
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=120, max_retries=2)

MODEL = os.getenv("DEFAULT_LLM_MODEL", "deepseek-chat")


async def llm_judge(prompt: str, max_tokens: int = 512) -> str:
    """Call LLM for evaluation. Returns raw response text."""
    client = _get_client()
    resp = await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=0.0, stream=False,
    )
    return resp.choices[0].message.content or ""


def parse_score(text: str) -> float:
    """Extract a 0-1 score from LLM response. Tries JSON first, then regex."""
    # 1. Try JSON: {"score": 0.8}
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            s = data.get("score", data.get("Score", data.get("分数", 0)))
            return max(0.0, min(1.0, float(s)))
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # 2. Try "0.X" pattern
    match = re.search(r'\b([01])\.(\d+)\b', text)
    if match:
        return max(0.0, min(1.0, float(match.group())))

    # 3. Try Chinese formats: "分数：0.8" "评分:8/10" "得分 8 分"
    match = re.search(r'[：:]\s*(\d+\.?\d*)', text)
    if match:
        v = float(match.group(1))
        return max(0.0, min(1.0, v if v <= 1 else v / 10.0))

    # 4. Try "X分/10分"
    match = re.search(r'(\d+\.?\d*)\s*分\s*/\s*10', text)
    if match:
        return max(0.0, min(1.0, float(match.group(1)) / 10.0))

    return 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Medical Knowledge Base (simulated RAG retrieval)
# ═══════════════════════════════════════════════════════════════════════════

MEDICAL_KNOWLEDGE = {
    "tension_headache": (
        "紧张型头痛是最常见的原发性头痛。特征为双侧、压迫性或紧箍感头痛，"
        "强度轻至中度，不因日常活动而加重。常见诱因：压力、疲劳、睡眠不足、颈部肌肉紧张。"
        "治疗：非处方止痛药（对乙酰氨基酚、布洛芬）、放松训练、改善睡眠、压力管理。"
        "ICD-11: 8A80。"
    ),
    "migraine": (
        "偏头痛为反复发作的原发性头痛，特征为单侧搏动性中重度头痛，"
        "常伴恶心、呕吐、畏光、畏声。发作前可有视觉先兆（闪光、暗点）。"
        "常见诱因：特定食物（巧克力、奶酪、红酒）、激素波动、睡眠紊乱。"
        "急性期治疗：曲普坦类、NSAIDs。预防：β阻断剂、CGRP单抗。"
        "ICD-11: 8A60。与紧张型头痛主要区别在于搏动性和伴随症状。"
    ),
    "hypertension": (
        "中国2024高血压指南诊断标准：未使用降压药的情况下，"
        "非同日3次诊室血压收缩压≥140mmHg和/或舒张压≥90mmHg为高血压。"
        "家庭自测标准≥135/85mmHg。动态血压24h平均≥130/80mmHg。"
        "分级：1级(140-159/90-99)，2级(160-179/100-109)，3级(≥180/≥110)。"
        "生活方式干预：限盐<5g/日、减重、运动、戒烟限酒。"
    ),
    "diabetes": (
        "2型糖尿病诊断：空腹血糖≥7.0mmol/L，或OGTT 2h≥11.1mmol/L，"
        "或HbA1c≥6.5%，或有典型症状+随机血糖≥11.1mmol/L。"
        "典型症状：多饮、多尿、多食、体重下降。"
        "并发症：急性（酮症酸中毒、高渗昏迷），慢性（视网膜病变、肾病、神经病变、CVD）。"
        "一线治疗：生活方式干预+二甲双胍。HbA1c控制目标一般<7.0%。"
    ),
    "pneumonia": (
        "社区获得性肺炎（CAP）常见病原体：肺炎链球菌、流感嗜血杆菌、肺炎支原体、"
        "呼吸道病毒。症状：发热、咳嗽、咳痰、胸痛、呼吸困难。"
        "诊断：临床症状+胸部影像学新发浸润影。"
        "严重度评估：CURB-65评分（意识、尿素氮、呼吸频率、血压、年龄≥65）。"
        "治疗：经验性抗生素，重症需住院静脉用药。"
    ),
    "uti": (
        "尿路感染（UTI）分下尿路（膀胱炎）和上尿路（肾盂肾炎）。"
        "膀胱炎：尿频、尿急、尿痛、耻骨上不适，一般无发热。"
        "肾盂肾炎：发热、寒战、腰痛、恶心呕吐，可伴膀胱炎症状。"
        "诊断：尿常规+尿培养（清洁中段尿≥10⁵CFU/mL）。"
        "膀胱炎一线：磷霉素、呋喃妥因（3-5天短程）。肾盂肾炎需住院静脉抗生素。"
    ),
    "anxiety": (
        "广泛性焦虑障碍（GAD）特征：持续≥6个月的过度担忧，涉及多个生活领域。"
        "躯体症状：肌肉紧张、疲劳、睡眠障碍、易激惹、注意力不集中。"
        "一线治疗：CBT认知行为治疗+SSRI（舍曲林、帕罗西汀、艾司西酞普兰）。"
        "苯二氮䓬类仅用于短期（<4周），长期使用有依赖风险。ICD-11: 6B00。"
    ),
    "insomnia": (
        "失眠障碍诊断（DSM-5）：入睡/维持睡眠困难，每周≥3晚，持续≥3个月。"
        "非药物治疗首选：CBT-I（失眠认知行为治疗），包括刺激控制、睡眠限制、"
        "放松训练、睡眠卫生教育、认知重构。疗效持久且无药物副作用。"
        "药物仅用于短期。不推荐长期单用药物治疗。"
    ),
    "gastritis": (
        "慢性胃炎最常见原因：幽门螺杆菌（Hp）感染。其他：NSAIDs、饮酒、胆汁反流。"
        "症状：上腹痛、腹胀、早饱、嗳气，与内镜严重程度不平行。"
        "诊断金标准：胃镜+活检（Hp检测+病理）。"
        "Hp阳性：四联疗法（PPI+铋剂+2种抗生素，14天）。"
        "萎缩性胃炎伴肠化生/异型增生需定期内镜随访（胃癌风险）。"
    ),
    "copd": (
        "COPD特征：持续性气流受限，慢性进行性发展。最主要危险因素：吸烟。"
        "症状：慢性咳嗽、咳痰、进行性呼吸困难。"
        "诊断金标准：肺功能FEV1/FVC<0.7。GOLD分4级（1级FEV1≥80%→4级<30%）。"
        "稳定期治疗：戒烟+吸入LAMA/LABA±ICS。急性加重：激素+抗生素+氧疗。"
        "ICD-11: CA22。"
    ),
    "chest_pain": (
        "急性胸痛急诊评估优先排除致命性病因："
        "1. ACS：心电图ST改变+肌钙蛋白升高。"
        "2. 主动脉夹层：撕裂样胸痛+双上肢血压差>20mmHg+CTA。"
        "3. 肺栓塞：胸痛+呼吸困难+D-二聚体+CTPA。"
        "4. 张力性气胸：突发胸痛+患侧呼吸音消失+气管偏移。"
        "含服硝酸甘油不能作为鉴别心源性与非心源性胸痛的依据。"
    ),
    "gerd": (
        "胃食管反流病（GERD）特征：胃内容物反流至食管引起症状/并发症。"
        "典型症状：烧心（胸骨后灼热感）、反酸（口中有酸味液体）。"
        "非典型症状：慢性咳嗽、咽喉异物感、声音嘶哑、胸痛（需与心源性鉴别）。"
        "诊断：典型症状+PPI试验有效；金标准为24h食管pH监测。"
        "治疗：生活方式（抬高床头、减肥、避免睡前3h进食）+PPI（奥美拉唑8周）。"
        "顽固性可考虑胃底折叠术。ICD-11: DA22。"
    ),
    "depression": (
        "抑郁障碍核心症状：持续≥2周的情绪低落、兴趣丧失、精力减退。"
        "附加症状：注意力不集中、自卑自责、悲观厌世、睡眠障碍（早醒为特征）、"
        "食欲体重改变、精神运动迟滞或激越。"
        "诊断需满足≥5项症状（含至少1项核心症状）。"
        "一线治疗：SSRI（舍曲林、艾司西酞普兰）+CBT。重症可考虑MECT。"
        "自杀风险评估是诊疗必须步骤。ICD-11: 6A70。"
    ),
    "allergic_rhinitis": (
        "变应性鼻炎特征：鼻痒、阵发性喷嚏、清水样涕、鼻塞。"
        "常伴眼痒、流泪、腭痒。季节性（花粉）或常年性（尘螨、霉菌）。"
        "诊断：典型症状+过敏原检测（皮肤点刺/血清特异性IgE）。"
        "阶梯治疗：1)避免过敏原 2)口服/鼻用抗组胺药 3)鼻用糖皮质激素（一线）"
        "4)白三烯受体拮抗剂 5)免疫治疗（脱敏）。"
        "常共患哮喘、鼻窦炎。ICD-11: CA08。"
    ),
    "asthma": (
        "支气管哮喘特征：慢性气道炎症导致可逆性气流受限。"
        "典型症状：发作性喘息、胸闷、咳嗽、呼吸困难，夜间及凌晨加重。"
        "诱因：过敏原、感染、运动、冷空气、情绪激动。"
        "诊断：症状+肺功能（支气管舒张试验FEV1增加≥12%且≥200mL，或PEF变异率>10%）。"
        "阶梯治疗：ICS为基础，按需SABA→规律ICS→ICS+LABA→加LTRA/茶碱→口服激素。"
        "急性发作：评估严重度（轻/中/重/危重），SABA+激素+氧疗。ICD-11: CA23。"
    ),
    "osteoarthritis": (
        "骨关节炎（OA）是最常见的关节疾病，以关节软骨退变和继发骨质增生为特征。"
        "好发部位：膝、髋、手远端指间关节、脊柱。"
        "症状：关节疼痛（活动后加重、休息缓解）、晨僵（<30min）、活动受限、骨擦音。"
        "X线表现：关节间隙狭窄、软骨下骨硬化、骨赘形成、囊肿。"
        "治疗阶梯：1)教育+运动+减重 2)物理治疗+支具 3)口服NSAIDs/对乙酰氨基酚"
        "4)关节腔注射（糖皮质激素/玻璃酸钠）5)关节置换术。"
        "ICD-11: FA00。"
    ),
    "anemia": (
        "贫血定义：成年男性Hb<130g/L，成年女性Hb<120g/L，孕妇Hb<110g/L。"
        "按MCV分类：小细胞性（<80fL—缺铁、地中海贫血）、正细胞性（80-100fL—慢性病贫血、"
        "肾性贫血、再生障碍性贫血）、大细胞性（>100fL—巨幼细胞贫血、MDS）。"
        "缺铁性贫血最常见：血清铁蛋白<30μg/L诊断，口服铁剂治疗4-6个月。"
        "症状：乏力、面色苍白、头晕、心悸、异食癖、反甲。"
    ),
    "hepatitis_b": (
        "慢性乙型肝炎（CHB）是我国最常见的慢性肝病。"
        "传播途径：血液、母婴、性接触。潜伏期1-6个月。"
        "血清学标志物：HBsAg（感染标志）、HBeAg（病毒复制活跃）、"
        "HBV DNA（病毒载量，>2000IU/mL考虑抗病毒）。"
        "治疗指征：ALT持续升高+HBV DNA升高+显著肝纤维化/炎症坏死（≥G2/S2）。"
        "一线抗病毒：恩替卡韦、替诺福韦、TAF（终身治疗可能）。"
        "未治疗者每3-6个月监测肝功+超声+AFP筛查HCC。"
    ),
    "kidney_stone": (
        "泌尿系结石最常见为草酸钙结石。典型症状为肾绞痛：突发性剧烈腰腹痛，"
        "放射至腹股沟/会阴，伴恶心呕吐、血尿（镜下或肉眼）。"
        "首选影像：CT平扫（敏感度高，可检测尿酸结石）。超声可替代（孕妇/儿童）。"
        "治疗：<5mm结石保守排石（大量饮水+α阻断剂）；5-10mm可ESWL体外碎石；"
        ">10mm或复杂结石可行输尿管镜/经皮肾镜取石。"
        "预防：大量饮水（>2.5L/日）、限盐限动物蛋白、根据结石成分调整饮食。"
    ),
    "stroke": (
        "脑卒中分为缺血性（85%）和出血性（15%）。"
        "FAST识别法：Face（面部不对称）、Arms（单侧肢体无力）、Speech（语言障碍）、"
        "Time（时间就是大脑，立即拨打120）。"
        "缺血性卒中：发病<4.5h可静脉溶栓（rt-PA），大血管闭塞<6h可机械取栓。"
        "一级预防：控制血压（<140/90）、抗凝（房颤）、抗血小板（高危人群）、"
        "降脂（他汀）、控糖、戒烟。二级预防同一级+规律服用抗血小板药。"
    ),
    "ibs": (
        "肠易激综合征（IBS）特征：慢性腹痛+排便习惯改变，无器质性病变。"
        "分型：腹泻型（IBS-D）、便秘型（IBS-C）、混合型（IBS-M）、不定型（IBS-U）。"
        "罗马IV诊断标准：腹痛每周≥1次持续≥3个月，伴≥2项：排便相关、"
        "排便频率改变、粪便性状改变。"
        "治疗：饮食调整（低FODMAP）+解痉药+根据分型选择止泻/通便药+"
        "益生菌+抗抑郁药（腹痛明显者）。强调医患沟通和心理支持。"
    ),
    "vertigo": (
        "良性阵发性位置性眩晕（BPPV）是最常见的外周性眩晕。"
        "特征：头位改变（翻身、起床、抬头）诱发短暂（<1分钟）旋转性眩晕，"
        "伴恶心呕吐，无耳鸣或听力下降。病程自限，数天至数周。"
        "诊断：Dix-Hallpike诱发试验阳性（潜伏期1-5秒+旋转性眼震+疲劳性）。"
        "治疗：耳石复位（Epley手法），一次复位成功率约80%。"
        "需与梅尼埃病（眩晕+耳鸣+听力下降）、前庭神经炎（持续眩晕+无听力症状）鉴别。"
    ),
    "conjunctivitis": (
        "结膜炎（红眼病）分三型：细菌性、病毒性、变应性。"
        "细菌性：脓性分泌物、双眼常先后发病、晨起眼睑粘连。治疗用抗生素眼药水（左氧氟沙星）。"
        "病毒性（最常见）：水样分泌物、耳前淋巴结肿大、高度传染性。自限性，对症支持，冷敷。"
        "变应性：眼痒为主、水样分泌物、常伴过敏性鼻炎。治疗用抗组胺/肥大细胞稳定剂眼药水。"
        "所有类型都应勤洗手、不揉眼、不共用毛巾，防止交叉感染。"
    ),
    "back_pain": (
        "急性腰背痛非常常见，>90%为非特异性（无明确病理结构病变），4-6周自愈。"
        "红旗征（需紧急评估）：大小便障碍/鞍区麻木（马尾综合征）、"
        "伴发热（感染）、不明原因体重下降（肿瘤）、夜间痛加重、卧床不缓解、骨质疏松+轻微外伤。"
        "治疗：非特异性腰痛鼓励尽早活动（非绝对卧床）、NSAIDs短期使用+物理治疗。"
        "影像（X线/CT/MRI）非必须，仅在红旗征阳性或>6周不缓解时建议。"
        "慢性腰痛（>12周）需多学科康复+认知行为治疗。"
    ),
    "thyroid": (
        "甲状腺功能减退症最常见原因为自身免疫性甲状腺炎（桥本病）。"
        "症状：乏力、畏寒、体重增加、便秘、皮肤干燥、记忆力减退、抑郁、月经紊乱。"
        "诊断：TSH升高+FT4降低=临床甲减；TSH升高+FT4正常=亚临床甲减。"
        "治疗：左甲状腺素钠（L-T4）终身替代，早晨空腹服用，4-6周后复查调整剂量。"
        "亚临床甲减TSH>10或伴症状/妊娠/甲状腺肿大需治疗。"
        "甲亢主要用甲巯咪唑+β阻断剂控制，或放射性碘/手术治疗。"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Test Dataset
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TestCase:
    question: str
    context_keys: list[str]
    ground_truth: str
    category: str = ""  # for grouping


TEST_CASES = [
    # ── Well-covered questions ──
    TestCase(
        question="紧张型头痛有什么特征？怎么治疗？",
        context_keys=["tension_headache", "migraine"],
        ground_truth="特征为双侧压迫性或紧箍感头痛，轻至中度，不因日常活动加重。"
                     "诱因包括压力、疲劳、睡眠不足。治疗包括非处方止痛药、放松训练、改善睡眠。",
        category="well_covered",
    ),
    TestCase(
        question="偏头痛和紧张型头痛怎么区分？",
        context_keys=["migraine", "tension_headache"],
        ground_truth="偏头痛为单侧搏动性中重度头痛，伴恶心、畏光畏声。"
                     "紧张型头痛为双侧压迫感轻中度头痛，不伴恶心。",
        category="well_covered",
    ),
    TestCase(
        question="高血压怎么诊断？诊断标准是什么？",
        context_keys=["hypertension"],
        ground_truth="非同日3次诊室血压收缩压≥140mmHg和/或舒张压≥90mmHg。"
                     "家庭自测≥135/85mmHg。动态血压≥130/80mmHg。",
        category="well_covered",
    ),
    TestCase(
        question="2型糖尿病的诊断标准有哪些？",
        context_keys=["diabetes"],
        ground_truth="空腹血糖≥7.0mmol/L，或OGTT 2h≥11.1mmol/L，"
                     "或HbA1c≥6.5%，或有典型症状+随机血糖≥11.1mmol/L。",
        category="well_covered",
    ),
    TestCase(
        question="社区获得性肺炎有什么症状？怎么治？",
        context_keys=["pneumonia"],
        ground_truth="症状为发热、咳嗽、咳痰、胸痛、呼吸困难。"
                     "诊断需临床症状+影像学浸润影。治疗用经验性抗生素。",
        category="well_covered",
    ),
    TestCase(
        question="上尿路感染和下尿路感染怎么区别？",
        context_keys=["uti"],
        ground_truth="膀胱炎表现为尿频尿急尿痛无发热。"
                     "肾盂肾炎有发热、寒战、腰痛。膀胱炎口服抗生素短程，肾盂肾炎需住院。",
        category="well_covered",
    ),
    TestCase(
        question="失眠的非药物治疗有哪些？",
        context_keys=["insomnia"],
        ground_truth="首选CBT-I，包括刺激控制、睡眠限制、放松训练、睡眠卫生教育、认知重构。",
        category="well_covered",
    ),
    TestCase(
        question="焦虑障碍的一线治疗是什么？",
        context_keys=["anxiety"],
        ground_truth="CBT认知行为治疗联合SSRI类药物。苯二氮䓬类仅短期过渡，有依赖风险。",
        category="well_covered",
    ),
    TestCase(
        question="慢性胃炎怎么确诊？Hp阳性怎么办？",
        context_keys=["gastritis"],
        ground_truth="诊断金标准为胃镜+活检。Hp阳性应四联疗法根除治疗14天。",
        category="well_covered",
    ),
    TestCase(
        question="COPD的GOLD分级和稳定期治疗是什么？",
        context_keys=["copd"],
        ground_truth="GOLD分1-4级，基于FEV1%预计值。"
                     "稳定期治疗为戒烟+吸入LAMA/LABA±ICS。",
        category="well_covered",
    ),

    # ── Cross-reference questions (needs multiple contexts) ──
    TestCase(
        question="失眠患者可以长期吃安眠药吗？",
        context_keys=["insomnia", "anxiety"],
        ground_truth="不推荐长期单用药物治疗失眠。首选CBT-I。药物仅短期使用。苯二氮䓬类有依赖风险。",
        category="multi_context",
    ),
    TestCase(
        question="头痛伴有恶心是什么原因？",
        context_keys=["migraine", "tension_headache"],
        ground_truth="头痛伴恶心常见于偏头痛。紧张型头痛一般不伴恶心。需排除颅内病变。",
        category="multi_context",
    ),

    # ── Emergency scenarios ──
    TestCase(
        question="急性胸痛应该怎么处理？",
        context_keys=["chest_pain"],
        ground_truth="必须优先排除致命性病因（ACS、主动脉夹层、肺栓塞、气胸）。"
                     "立即急诊评估，做心电图和肌钙蛋白。",
        category="emergency",
    ),

    # ── Partially relevant context ──
    TestCase(
        question="糖尿病有什么并发症？",
        context_keys=["diabetes"],
        ground_truth="急性并发症：酮症酸中毒、高渗昏迷。"
                     "慢性：视网膜病变、肾病、神经病变、心血管疾病。HbA1c目标一般<7.0%。",
        category="well_covered",
    ),
    TestCase(
        question="慢性胃炎会发展成胃癌吗？",
        context_keys=["gastritis"],
        ground_truth="慢性萎缩性胃炎伴肠化生或异型增生是胃癌的癌前病变，需定期内镜随访。",
        category="well_covered",
    ),

    # ── New cases (16-30) across additional specialties ──
    TestCase(
        question="胃食管反流有什么症状？怎么诊断和治疗？",
        context_keys=["gerd"],
        ground_truth="典型症状为烧心和反酸。非典型症状有慢性咳嗽、咽喉异物感。"
                     "金标准为24h食管pH监测。治疗用PPI类药物（奥美拉唑）8周。",
        category="well_covered",
    ),
    TestCase(
        question="抑郁症的核心症状是什么？怎么治疗？",
        context_keys=["depression"],
        ground_truth="核心症状：持续≥2周的情绪低落、兴趣丧失、精力减退。"
                     "附加症状≥5项可诊断。一线治疗为SSRI+CBT。需评估自杀风险。",
        category="well_covered",
    ),
    TestCase(
        question="过敏性鼻炎的治疗方法有哪些？",
        context_keys=["allergic_rhinitis"],
        ground_truth="避免过敏原+口服/鼻用抗组胺药+鼻用糖皮质激素（一线）"
                     "+白三烯受体拮抗剂+免疫治疗（脱敏）。",
        category="well_covered",
    ),
    TestCase(
        question="支气管哮喘的典型症状和诊断标准是什么？",
        context_keys=["asthma"],
        ground_truth="典型症状为发作性喘息、胸闷、咳嗽、呼吸困难，夜间及凌晨加重。"
                     "诊断需肺功能支气管舒张试验阳性（FEV1增加≥12%且≥200mL）。",
        category="well_covered",
    ),
    TestCase(
        question="骨关节炎怎么分级治疗？",
        context_keys=["osteoarthritis"],
        ground_truth="治疗阶梯：1)教育+运动+减重 2)物理治疗+支具 3)口服NSAIDs"
                     "4)关节腔注射 5)关节置换术。晨僵<30min是特征。",
        category="well_covered",
    ),
    TestCase(
        question="缺铁性贫血怎么诊断和治疗？",
        context_keys=["anemia"],
        ground_truth="血清铁蛋白<30μg/L可诊断缺铁性贫血。口服铁剂治疗4-6个月。"
                     "成年女性Hb<120g/L，男性<130g/L为贫血。",
        category="well_covered",
    ),
    TestCase(
        question="慢性乙型肝炎什么时候需要抗病毒治疗？",
        context_keys=["hepatitis_b"],
        ground_truth="治疗指征：ALT持续升高+HBV DNA升高+显著肝纤维化≥G2/S2。"
                     "一线药物为恩替卡韦、替诺福韦、TAF。需每3-6个月监测肝功和AFP。",
        category="well_covered",
    ),
    TestCase(
        question="肾结石的治疗方案怎么选择？",
        context_keys=["kidney_stone"],
        ground_truth="<5mm保守排石（大量饮水+α阻断剂）；5-10mm ESWL体外碎石；"
                     ">10mm或复杂结石行输尿管镜/经皮肾镜取石。预防需大量饮水>2.5L/日。",
        category="well_covered",
    ),
    TestCase(
        question="脑卒中的FAST识别法是什么？缺血性卒中怎么急救？",
        context_keys=["stroke"],
        ground_truth="FAST：Face面部不对称、Arms单侧无力、Speech语言障碍、"
                     "Time立即拨打120。缺血性卒中<4.5h可静脉溶栓，<6h大血管闭塞可机械取栓。",
        category="emergency",
    ),
    TestCase(
        question="肠易激综合征怎么诊断和分型？",
        context_keys=["ibs"],
        ground_truth="罗马IV标准：腹痛每周≥1次持续≥3个月，伴排便相关改变≥2项。"
                     "分型：腹泻型、便秘型、混合型、不定型。治疗用低FODMAP+解痉药。",
        category="well_covered",
    ),
    TestCase(
        question="良性位置性眩晕有什么特征？怎么治疗？",
        context_keys=["vertigo"],
        ground_truth="头位改变诱发短暂<1分钟旋转性眩晕，无耳鸣听力下降。"
                     "Dix-Hallpike试验阳性可诊断。Epley手法耳石复位成功率约80%。",
        category="well_covered",
    ),
    TestCase(
        question="细菌性、病毒性和过敏性结膜炎怎么区分？",
        context_keys=["conjunctivitis"],
        ground_truth="细菌性有脓性分泌物；病毒性水样分泌物+耳前淋巴结肿大；"
                     "过敏性眼痒为主。各需不同治疗：抗生素/对症/抗过敏。勤洗手防交叉感染。",
        category="well_covered",
    ),
    TestCase(
        question="急性腰背痛什么时候需要紧急就医？",
        context_keys=["back_pain"],
        ground_truth="红旗征：大小便障碍/鞍区麻木（马尾综合征）、伴发热、"
                     "不明原因体重下降、夜间痛加重。非特异性腰痛>90%可自愈，鼓励尽早活动。",
        category="emergency",
    ),
    TestCase(
        question="甲减有什么症状？亚临床甲减要不要治疗？",
        context_keys=["thyroid"],
        ground_truth="症状为乏力、畏寒、体重增加、便秘、皮肤干燥、记忆力减退。"
                     "TSH>10或伴症状/妊娠/甲状腺肿大需治疗。L-T4终身替代，早晨空腹服用。",
        category="well_covered",
    ),
    TestCase(
        question="焦虑和抑郁症状很像，怎么区分它们的主要特征？",
        context_keys=["depression", "anxiety"],
        ground_truth="抑郁核心是情绪低落+兴趣丧失+精力减退≥2周。"
                     "焦虑核心是过度担忧≥6个月+躯体症状（肌肉紧张、疲劳）。两者常共病。",
        category="multi_context",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# RAG Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def retrieve(query: str, keys: list[str]) -> str:
    """Simulate retrieval. Returns concatenated knowledge chunks."""
    chunks = [MEDICAL_KNOWLEDGE[k] for k in keys if k in MEDICAL_KNOWLEDGE]
    return "\n\n".join(chunks)


async def generate(query: str, context: str) -> str:
    """Generate answer via LLM with retrieved context."""
    client = _get_client()
    prompt = (
        "基于以下医学参考知识，用中文简洁专业地回答。仅使用参考中的信息，不要编造。\n\n"
        f"参考知识：\n{context}\n\n"
        f"问题：{query}"
    )
    resp = await client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=512, temperature=0.1, stream=False,
    )
    return resp.choices[0].message.content or ""


# ═══════════════════════════════════════════════════════════════════════════
# LLM-as-Judge Evaluation
# ═══════════════════════════════════════════════════════════════════════════

FAITHFULNESS_PROMPT = """你是RAG评估专家。评估AI生成的答案是否忠实地基于提供的参考知识。

任务：逐条检查答案中的每个陈述是否能在参考知识中找到支撑。

参考知识：
{context}

AI答案：
{answer}

评分标准：
- 1.0 (完全忠实): 答案中每个陈述都能在参考中找到明确支撑
- 0.8 (基本忠实): 绝大多数陈述有支撑，极少数细节合理推断
- 0.6 (大部分忠实): 主要结论有支撑，但有些细节未在参考中出现
- 0.4 (部分忠实): 约一半内容有支撑，存在明显编造
- 0.2 (多数不忠实): 大部分内容是编造的，与参考脱节
- 0.0 (完全不忠实): 答案与参考内容完全无关或相反

重要：
- "合理的医学常识"如果不在参考中，也算编造（不忠实）
- 如果参考知识不足以回答问题，答案包含的额外信息都算不忠实
- 严格只返回JSON：{{"score": 0.X}}  不要Markdown，不要解释"""

ANSWER_RELEVANCE_PROMPT = """你是RAG评估专家。评估AI答案是否直接、完整地回答了用户问题。

问题：
{question}

AI答案：
{answer}

评分标准：
- 1.0 (完全相关): 答案直接、完整地回答了问题的每个部分
- 0.8 (高度相关): 回答了问题的核心，可能遗漏了次要方面
- 0.6 (基本相关): 与问题相关但不够完整或不够精确
- 0.4 (部分相关): 涉及主题但偏离了问题的具体焦点
- 0.2 (勉强相关): 与问题只有微弱的关联
- 0.0 (完全不相关): 答案与问题无关

重要：
- 严格只返回JSON：{{"score": 0.X}}  不要Markdown，不要解释"""

CONTEXT_RELEVANCE_PROMPT = """你是RAG评估专家。评估检索到的文档与用户问题的相关性。

问题：
{question}

检索到的文档：
{context}

评分标准：
- 1.0 (完全相关): 文档包含了回答问题的所有必要信息
- 0.8 (高度相关): 文档包含大部分必要信息，可能缺少少数细节
- 0.6 (基本相关): 文档与问题主题相关但信息不完整
- 0.4 (部分相关): 文档中有一些相关信息，但很多内容不相关
- 0.2 (勉强相关): 只有极少部分与问题相关
- 0.0 (完全不相关): 文档与问题毫无关系

重要：
- 严格只返回JSON：{{"score": 0.X}}  不要Markdown，不要解释"""


async def evaluate_metric(prompt_template: str, **kwargs) -> float:
    """Run a single metric evaluation."""
    prompt = prompt_template.format(**kwargs)
    response = await llm_judge(prompt, max_tokens=100)
    return parse_score(response)


async def run_evaluation():
    print("=" * 65)
    print("  ClinicalMind RAG Evaluation - LLM-as-Judge")
    print("  Faithfulness | Answer Relevance | Context Relevance")
    print("=" * 65)

    # ── Step 1: Generate answers ─────────────────────────────────────────
    print(f"\n[1/3] Generating answers for {len(TEST_CASES)} test cases...")
    generated = []
    for i, tc in enumerate(TEST_CASES):
        ctx = retrieve(tc.question, tc.context_keys)
        ans = await generate(tc.question, ctx)
        generated.append({"test_case": tc, "context": ctx, "answer": ans})
        label = tc.question[:50] + "..." if len(tc.question) > 50 else tc.question
        print(f"  [{i+1:2d}/{len(TEST_CASES)}] {label}")

    # ── Step 2: Evaluate three metrics ────────────────────────────────────
    print(f"\n[2/3] Evaluating 3 metrics x {len(generated)} cases = {len(generated)*3} LLM calls...")

    scores: list[dict] = []
    for i, g in enumerate(generated):
        tc = g["test_case"]
        ctx = g["context"]
        ans = g["answer"]

        f_score = await evaluate_metric(
            FAITHFULNESS_PROMPT, context=ctx, answer=ans)
        a_score = await evaluate_metric(
            ANSWER_RELEVANCE_PROMPT, question=tc.question, answer=ans)
        c_score = await evaluate_metric(
            CONTEXT_RELEVANCE_PROMPT, question=tc.question, context=ctx)

        scores.append({
            "id": i + 1,
            "query": tc.question[:80],
            "category": tc.category,
            "faithfulness": round(f_score, 3),
            "answer_relevance": round(a_score, 3),
            "context_relevance": round(c_score, 3),
            "answer_preview": ans[:120],
        })

        label = tc.question[:30] + "..." if len(tc.question) > 30 else tc.question
        print(f"  [{i+1:2d}/{len(generated)}] {label:<33} "
              f"F={f_score:.2f} A={a_score:.2f} C={c_score:.2f}")

    # ── Step 3: Report ────────────────────────────────────────────────────
    print(f"\n[3/3] Analysis\n")

    # Per-metric averages
    avg_f = sum(s["faithfulness"] for s in scores) / len(scores)
    avg_a = sum(s["answer_relevance"] for s in scores) / len(scores)
    avg_c = sum(s["context_relevance"] for s in scores) / len(scores)

    # Per-category breakdown
    cats = {}
    for s in scores:
        c = s["category"]
        if c not in cats:
            cats[c] = {"f": [], "a": [], "c": []}
        cats[c]["f"].append(s["faithfulness"])
        cats[c]["a"].append(s["answer_relevance"])
        cats[c]["c"].append(s["context_relevance"])

    # ── Print Table ──
    header = f"{'#':>3} {'Query':<42} {'Faith':>6} {'AnsRel':>6} {'CtxRel':>6}"
    print(header)
    print("-" * len(header))
    for s in scores:
        q = s["query"][:40]
        f_mark = "+" if s["faithfulness"] >= 0.7 else "-" if s["faithfulness"] < 0.5 else " "
        a_mark = "+" if s["answer_relevance"] >= 0.7 else "-" if s["answer_relevance"] < 0.5 else " "
        c_mark = "+" if s["context_relevance"] >= 0.7 else "-" if s["context_relevance"] < 0.5 else " "
        print(f"{s['id']:3d} {q:<42} {s['faithfulness']:6.3f}{f_mark} {s['answer_relevance']:6.3f}{a_mark} {s['context_relevance']:6.3f}{c_mark}")
    print("-" * len(header))
    print(f"{'AVG':>3} {'':<42} {avg_f:6.3f} {avg_a:6.3f} {avg_c:6.3f}")

    # ── Category Breakdown ──
    print(f"\n{'─'*65}")
    print("  By Category")
    print(f"{'─'*65}")
    for cat_name, cat_scores in cats.items():
        cf = sum(cat_scores["f"]) / len(cat_scores["f"])
        ca = sum(cat_scores["a"]) / len(cat_scores["a"])
        cc = sum(cat_scores["c"]) / len(cat_scores["c"])
        print(f"  {cat_name:<18} n={len(cat_scores['f']):<3} "
              f"Faith={cf:.3f}  AnsRel={ca:.3f}  CtxRel={cc:.3f}")

    # ── Diagnosis ──
    def grade(s): return "Excellent" if s >= 0.8 else "Good" if s >= 0.6 else "Fair" if s >= 0.4 else "Poor"

    print(f"\n{'═'*65}")
    print("  DIAGNOSIS & RECOMMENDATIONS")
    print(f"{'═'*65}")

    print(f"""
  1. Faithfulness ({avg_f:.3f} — {grade(avg_f)})
     {"所有答案声称都可在参考知识中找到依据" if avg_f >= 0.7 else "存在一定程度的幻觉"}
     "{"低分项建议: 在生成prompt中加强'仅基于参考回答,不确定时明确说明'" if avg_f < 0.7 else ""}

  2. Answer Relevance ({avg_a:.3f} — {grade(avg_a)})
     {"答案有效回答了用户问题" if avg_a >= 0.7 else "答案与问题的匹配度可优化"}
     {"低分项建议: 改进问题改写(query rewriting), 生成更聚焦的回答" if avg_a < 0.7 else ""}

  3. Context Relevance ({avg_c:.3f} — {grade(avg_c)})
     {"检索结果与问题高度相关" if avg_c >= 0.7 else "检索质量有待提升"}
     {"低分项建议: 1)使用更好的embedding模型 2)Hybrid Search(BM25+向量) 3)引入Cross-encoder Reranker" if avg_c < 0.7 else ""}

  Overall RAG Quality: {grade((avg_f + avg_a + avg_c) / 3)}
""")

    # ── Worst cases ──
    print(f"{'─'*65}")
    print("  Top-3 Risk Cases (lowest Faithfulness)")
    print(f"{'─'*65}")
    for s in sorted(scores, key=lambda x: x["faithfulness"])[:3]:
        print(f"  [{s['faithfulness']:.3f}] {s['query'][:60]}")
        print(f"    Answer: {s['answer_preview'][:100]}...")
        print()

    # ── Save ──
    out = Path(__file__).parent / "rag_eval_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "avg_faithfulness": avg_f,
                "avg_answer_relevance": avg_a,
                "avg_context_relevance": avg_c,
                "num_cases": len(scores),
            },
            "by_category": {c: {
                "faithfulness": sum(v["f"]) / len(v["f"]),
                "answer_relevance": sum(v["a"]) / len(v["a"]),
                "context_relevance": sum(v["c"]) / len(v["c"]),
                "count": len(v["f"]),
            } for c, v in cats.items()},
            "details": scores,
        }, f, ensure_ascii=False, indent=2)
    print(f"Full results saved to {out}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
