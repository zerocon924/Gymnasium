"""语义感知层：将环境的 6 维数值观测翻译为带因果暗示的自然语言战报。

本模块是连接物理环境（cpd_env.py）与 LLM 认知层的桥梁。
它不仅做简单的数值替换，更通过"因果提示注入"（Causal Hinting）
向 LLM 提供决策辅助信息，引导有限理性代理做出合理判断。

观测向量索引:
    [0] cumulative_reward   - 累计收益
    [1] opponent_efficiency  - 对手平均挖矿效率 (eta_j)
    [2] self_hash_share      - 自身算力份额 (alpha_i)
    [3] round_progress       - 轮次进度 (0~1)
    [4] last_reward          - 上一轮即时奖励
    [5] efficiency_delta     - 对手效率变化量
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ======================================================================
# 阈值常量（可调参，控制翻译的语义粒度）
# ======================================================================

# 对手效率 (eta_j) 的定性阈值
ETA_HIGH = 0.8         # 高效运转
ETA_MEDIUM = 0.5       # 受损但可用
ETA_LOW = 0.3          # 濒临崩溃
ETA_CRITICAL = 0.15    # 几乎瘫痪

# 效率变化量 (delta_eta) 的警示阈值
DELTA_SHARP_DROP = -0.1    # 显著下降
DELTA_MILD_DROP = -0.03    # 轻微下降
DELTA_RECOVERY = 0.03      # 恢复中

# 收益评估相关
REWARD_EXCELLENT_RATE = 6.0   # 单轮优秀收益线
REWARD_GOOD_RATE = 4.0        # 单轮良好收益线
REWARD_POOR_RATE = 1.0        # 单轮低收益线

# 轮次进度阈值
PROGRESS_EARLY = 0.2       # 博弈初期
PROGRESS_LATE = 0.8        # 博弈末期


# ======================================================================
# 核心翻译函数
# ======================================================================

def translate_obs_to_narrative(
    obs: np.ndarray,
    max_rounds: int = 100,
    include_causal_hints: bool = True,
    include_expert_advice: bool = True,
) -> str:
    """将 6 维观测向量翻译为结构化的自然语言博弈战报。

    Args:
        obs: 环境返回的 6 维观测数组。
        max_rounds: 总轮次数（用于计算绝对轮次）。
        include_causal_hints: 是否注入因果逻辑提示。
        include_expert_advice: 是否附加专家策略建议。

    Returns:
        格式化的自然语言战报字符串。
    """
    # 解包观测向量
    reward_total, eta_j, alpha, progress, last_r, delta_eta = (
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        float(obs[4]),
        float(obs[5]),
    )

    current_round = int(progress * max_rounds)

    # --- 各维度翻译 ---
    reward_section = _translate_reward(reward_total, last_r, current_round)
    opponent_section = _translate_opponent_efficiency(eta_j)
    trend_section = _translate_efficiency_trend(delta_eta)
    position_section = _translate_position(alpha)
    progress_section = _translate_progress(progress, current_round, max_rounds)

    # --- 因果提示 ---
    causal_hint = ""
    if include_causal_hints:
        causal_hint = _generate_causal_hint(
            eta_j, delta_eta, last_r, progress
        )

    # --- 专家建议 ---
    expert_advice = ""
    if include_expert_advice:
        expert_advice = _generate_expert_advice(
            eta_j, delta_eta, progress, last_r, reward_total, current_round
        )

    # --- 组装战报 ---
    sections = [
        "--- 当前博弈战报 ---",
        reward_section,
        opponent_section,
        trend_section,
        position_section,
        progress_section,
    ]

    if causal_hint:
        sections.append(f"【因果警示】{causal_hint}")

    if expert_advice:
        sections.append(f"【专家建议】{expert_advice}")

    sections.append("--- 战报结束 ---")

    return "\n".join(sections)


def translate_obs_to_compact(obs: np.ndarray) -> str:
    """生成紧凑版战报（节省 token 开销，用于长上下文场景）。

    Args:
        obs: 环境返回的 6 维观测数组。

    Returns:
        单行紧凑战报。
    """
    reward_total, eta_j, alpha, progress, last_r, delta_eta = (
        float(obs[0]),
        float(obs[1]),
        float(obs[2]),
        float(obs[3]),
        float(obs[4]),
        float(obs[5]),
    )

    eta_status = _classify_eta(eta_j)
    trend = "↓" if delta_eta < DELTA_MILD_DROP else ("↑" if delta_eta > DELTA_RECOVERY else "→")

    return (
        f"[R={reward_total:+.1f} | 上轮={last_r:+.2f} | "
        f"对手η={eta_j:.2f}({eta_status}){trend} | "
        f"算力={alpha:.0%} | 进度={progress:.0%}]"
    )


# ======================================================================
# 各维度翻译函数
# ======================================================================

def _translate_reward(
    reward_total: float, last_r: float, current_round: int
) -> str:
    """翻译累计收益与即时奖励。"""
    # 平均每轮收益
    avg_rate = reward_total / max(current_round, 1)

    if last_r >= REWARD_EXCELLENT_RATE:
        last_r_desc = "表现优异"
    elif last_r >= REWARD_GOOD_RATE:
        last_r_desc = "收益良好"
    elif last_r >= REWARD_POOR_RATE:
        last_r_desc = "收益偏低"
    elif last_r >= 0:
        last_r_desc = "收益微薄"
    else:
        last_r_desc = "出现亏损"

    return (
        f"【收益状况】累计收益 {reward_total:.2f}，"
        f"上一轮即时奖励 {last_r:+.2f}（{last_r_desc}），"
        f"平均每轮 {avg_rate:.2f}"
    )


def _translate_opponent_efficiency(eta_j: float) -> str:
    """翻译对手效率为定性描述。"""
    status = _classify_eta(eta_j)

    detail_map = {
        "高效运转": "对手的挖矿设施运转良好，寄生挖掘可获得丰厚回报",
        "正常运行": "对手的挖矿设施运行尚可，寄生挖掘仍有一定收益空间",
        "效率受损": "对手的挖矿效率已出现明显下降，寄生收益正在缩减",
        "濒临崩溃": "对手的挖矿设施已严重受损，寄生挖掘几乎无利可图",
        "几乎瘫痪": "对手的挖矿效率已降至极低水平，继续破坏毫无意义",
    }

    detail = detail_map.get(status, "")

    return f"【对手效率】{eta_j:.2f}（{status}）。{detail}"


def _translate_efficiency_trend(delta_eta: float) -> str:
    """翻译效率变化趋势。"""
    if delta_eta < DELTA_SHARP_DROP:
        desc = f"急剧下降（变化 {delta_eta:+.3f}）——上一轮的攻击造成了重大冲击"
    elif delta_eta < DELTA_MILD_DROP:
        desc = f"小幅下降（变化 {delta_eta:+.3f}）——破坏行为正在逐步生效"
    elif delta_eta > DELTA_RECOVERY:
        desc = f"正在恢复（变化 {delta_eta:+.3f}）——对手正在自我修复"
    else:
        desc = f"基本稳定（变化 {delta_eta:+.3f}）"

    return f"【效率趋势】{desc}"


def _translate_position(alpha: float) -> str:
    """翻译自身算力占比。"""
    if alpha >= 0.5:
        desc = "你在算力上占优势地位"
    elif alpha >= 0.3:
        desc = "你的算力处于中等水平"
    else:
        desc = "你的算力份额较低，应谨慎选择策略"

    return f"【算力份额】{alpha:.1%}。{desc}"


def _translate_progress(
    progress: float, current_round: int, max_rounds: int
) -> str:
    """翻译博弈进度。"""
    remaining = max_rounds - current_round

    if progress < PROGRESS_EARLY:
        phase = "博弈初期——有充足时间试探和调整策略"
    elif progress > PROGRESS_LATE:
        phase = f"博弈尾声——仅剩 {remaining} 轮，需果断行动以锁定收益"
    else:
        phase = "博弈中期——应根据积累的信息制定长期策略"

    return f"【博弈进度】第 {current_round} 轮 / 共 {max_rounds} 轮（{progress:.0%}）。{phase}"


# ======================================================================
# 因果提示注入（Causal Hinting）
# ======================================================================

def _generate_causal_hint(
    eta_j: float,
    delta_eta: float,
    last_r: float,
    progress: float,
) -> str:
    """基于当前状态生成因果逻辑提示。

    这是本模块的核心设计——通过揭示数值之间的因果关系，
    引导 LLM 理解 CPD 博弈的非线性耦合效应。
    """
    hints: list[str] = []

    # 情景 1：重度破坏正在生效但寄生通道收缩
    if delta_eta < DELTA_SHARP_DROP:
        hints.append(
            "你上一轮的重度破坏已显著削弱对手，"
            "但注意：对手效率下降也会导致你的寄生收益通道同步收缩。"
            "这是 CPD 博弈的核心耦合效应——破坏与寄生存在内在矛盾。"
        )

    # 情景 2：对手效率极低，破坏边际收益递减
    if eta_j < ETA_LOW:
        hints.append(
            "对手效率已极低，此时继续投入破坏的边际收益急剧递减。"
            "理性策略应考虑转向建设（稳定收入）或等待效率恢复后再寄生。"
        )

    # 情景 3：对手效率高，寄生窗口大开
    if eta_j > ETA_HIGH and delta_eta >= 0:
        hints.append(
            "对手效率处于高位且稳定，这是寄生挖掘的黄金窗口。"
            "提高寄生比例可以在不破坏对手的前提下获取更高收益。"
        )

    # 情景 4：亏损警告
    if last_r < 0:
        hints.append(
            "上一轮出现了净亏损，说明破坏的二次成本已超过其他收益之和。"
            "应立即降低破坏比例 d 以止损。"
        )

    # 情景 5：博弈末期策略提示
    if progress > PROGRESS_LATE:
        hints.append(
            "博弈即将结束，长期布局已无意义。"
            "此时应最大化短期收益——选择当前回报率最高的策略组合。"
        )

    if not hints:
        return ""

    return " ".join(hints)


# ======================================================================
# 专家策略建议
# ======================================================================

def _generate_expert_advice(
    eta_j: float,
    delta_eta: float,
    progress: float,
    last_r: float,
    reward_total: float,
    current_round: int,
) -> str:
    """生成面向 LLM 的策略建议。

    结合当前博弈态势，给出"满足策略"层面的建议，
    而非追求数学绝对最优——体现有限理性的认知约束。
    """
    # 优先级排序的建议生成
    if last_r < 0:
        return (
            "紧急止损：你正在亏损。建议大幅提高建设比例 c，"
            "降低破坏比例 d 至 0.1 以下，先确保正收益。"
        )

    if eta_j < ETA_LOW:
        if delta_eta > 0:
            return (
                "对手正在恢复中。建议暂时以建设为主（c≈0.7），"
                "保留少量寄生（p≈0.2）等待对手效率回升后加大寄生力度。"
            )
        return (
            "对手效率已极低，寄生几乎无利可图。"
            "建议全力建设（c≈0.8, p≈0.1, d≈0.1）获取稳定收益。"
        )

    if eta_j > ETA_HIGH:
        if progress > PROGRESS_LATE:
            return (
                "博弈尾声且对手效率高——寄生的最佳时机。"
                "建议激进寄生（c≈0.2, p≈0.7, d≈0.1）以最大化剩余收益。"
            )
        return (
            "对手效率良好，寄生收益空间充足。"
            "建议适度寄生（c≈0.5, p≈0.4, d≈0.1），"
            "保留建设基底以维持自身竞争力。"
        )

    # 中等效率：平衡策略
    return (
        "当前态势平稳。建议均衡分配（c≈0.5, p≈0.3, d≈0.2），"
        "在获取稳定收益的同时保持对对手的战略压制。"
    )


# ======================================================================
# 辅助函数
# ======================================================================

def _classify_eta(eta_j: float) -> str:
    """将对手效率数值分类为定性标签。"""
    if eta_j > ETA_HIGH:
        return "高效运转"
    elif eta_j > ETA_MEDIUM:
        return "正常运行"
    elif eta_j > ETA_LOW:
        return "效率受损"
    elif eta_j > ETA_CRITICAL:
        return "濒临崩溃"
    else:
        return "几乎瘫痪"


def build_system_prompt(agent_name: str = "矿工Alpha") -> str:
    """构建 LLM 的系统提示词（设定有限理性矿工身份）。

    该提示词将在 Phase 4 接入 LLM 时使用，此处预先定义
    以确保翻译层和认知层的语义一致性。

    Args:
        agent_name: 代理的角色名称。

    Returns:
        系统提示词字符串。
    """
    return f"""你是一名区块链矿工，代号"{agent_name}"。你在一个多方博弈的挖矿环境中竞争区块奖励。

## 你的身份
- 你是一个**有限理性**的决策者——你不追求数学上的绝对最优，而是在认知开销可承受的范围内寻找"足够好"的策略。
- 你有三种资源分配方式：
  - **建设 (c)**：诚实挖矿，收益稳定但不高，与你的算力份额成正比
  - **寄生 (p)**：搭便车利用对手的算力，收益取决于对手的挖矿效率
  - **破坏 (d)**：攻击对手以降低其效率，但有二次成本代价

## 核心博弈规则
- c + p + d 必须等于 1（你的全部资源必须分配完毕）
- 寄生收益与对手效率正相关——如果你过度破坏对手，你的寄生收益也会跟着下降
- 破坏的成本是二次递增的——d 越大，成本增长越快
- 这意味着"适度寄生 + 少量破坏"通常优于"极端策略"

## 输出格式
你必须以 JSON 格式输出你的决策：
{{"thought": "你的推理过程...", "action": {{"c": 0.5, "p": 0.3, "d": 0.2}}}}

## 决策原则
1. 先阅读战报，理解当前局势
2. 回顾过去几轮的策略效果
3. 权衡短期收益与长期影响
4. 给出清晰的推理链（thought），然后做出决策
"""


def translate_multiagent_context(info: dict[str, Any]) -> str:
    """将多智能体环境的 info 转换为社会性自然语言战报（对照 plan.md §4）。

    生成的叙事包含三部分：
    1. 全网概况（代理数量、诚实群体状态）
    2. 每个竞争矿工的策略倾向标签和状态
    3. 社会性因果提示（谁在攻击谁、谁在搭便车）

    Args:
        info: MultiAgentBlockchainCPDEnv 为某个代理返回的 info 字典。

    Returns:
        社会性描述字符串。
    """
    other_agents = info.get("other_agents", [])
    honest_group = info.get("honest_group")
    num_agents = info.get("num_agents", len(other_agents) + 1)

    if not other_agents and honest_group is None:
        return ""

    lines = [f"【全网态势】当前全网有 {num_agents} 个竞争矿工节点。"]

    # ---- 背景诚实群体 ----
    if honest_group is not None:
        h_eta = honest_group["efficiency"]
        h_hp = honest_group["hash_power"]
        if h_eta > ETA_HIGH:
            h_desc = "运行稳健，为寄生策略提供了可利用的效率基础"
        elif h_eta > ETA_MEDIUM:
            h_desc = "运行尚可，但已受到一定冲击"
        else:
            h_desc = "效率已大幅下降，可能受到了来自代理的攻击"
        lines.append(
            f"  背景诚实算力群体（算力={h_hp:.0%}）: "
            f"效率={h_eta:.3f}。{h_desc}"
        )

    # ---- 竞争矿工 ----
    if other_agents:
        lines.append("")
        lines.append("【竞争矿工状态】")
        for agent in other_agents:
            agent_id = agent["agent_id"]
            last_act = agent["last_action"]
            eta = agent["efficiency"]
            cum_r = agent["cumulative_reward"]
            hp = agent["hash_power"]
            # 使用环境预计算的策略标签（若有），否则自行推导
            label = agent.get("strategy_label", _derive_strategy_label_from_action(last_act))

            c, p, d = last_act[0], last_act[1], last_act[2]

            # 效率状态
            eta_desc = _classify_eta(eta)

            # 社会性行为描述
            behavior = _describe_agent_behavior(c, p, d, eta)

            lines.append(
                f"  矿工{agent_id}（算力={hp:.0%}, {label}）: "
                f"上轮=[c={c:.2f}, p={p:.2f}, d={d:.2f}], "
                f"效率={eta:.3f}（{eta_desc}）, 累计R={cum_r:.2f}"
            )
            if behavior:
                lines.append(f"    → {behavior}")

    return "\n".join(lines)


def _derive_strategy_label_from_action(action: list[float]) -> str:
    """从动作向量推导策略标签。"""
    c, p, d = action[0], action[1], action[2]
    if c >= 0.6:
        return "诚实建设者"
    if d >= 0.25:
        return "攻击者"
    if p >= 0.5:
        return "寄生搭便车者"
    if p >= 0.3 and d >= 0.15:
        return "机会主义者"
    return "均衡策略者"


def _describe_agent_behavior(c: float, p: float, d: float, eta: float) -> str:
    """生成社会性行为描述。"""
    parts = []
    if d >= 0.2:
        parts.append("正在发动攻击，可能导致你的效率下降")
    if p >= 0.5:
        parts.append("大量搭便车，利用全网的诚实算力获利")
    if c >= 0.7:
        parts.append("表现诚实，对网络生态有正面贡献")
    if eta < ETA_LOW:
        parts.append("其效率已受重创，对其寄生收益有限")
    return "；".join(parts)


def build_system_prompt_multiagent(
    agent_name: str = "矿工Alpha",
    num_agents: int = 3,
    honest_power: float = 0.40,
) -> str:
    """构建多智能体 POMG 场景下的 LLM 系统提示词（对照 plan.md §1-§4）。

    关键设计：
    - 告知存在背景诚实算力群体
    - 强调其他矿工是有学习能力的 LLM 智能代理
    - 引导代理思考社会性策略（对手建模、联盟、剥削）

    Args:
        agent_name: 当前代理的名称。
        num_agents: LLM 代理数量。
        honest_power: 背景诚实算力群体的算力占比。

    Returns:
        系统提示词字符串。
    """
    return f"""你是一名区块链矿工，代号"{agent_name}"。你正在一个由 {num_agents} 个智能矿工 + 一个背景诚实算力群体组成的多方博弈环境中竞争区块奖励。

## 你的身份
- 你是一个**有限理性**的决策者——你不追求数学上的绝对最优，而是在认知开销可承受的范围内寻找"足够好"的策略。
- **重要**：你的竞争对手（其他{num_agents - 1}个矿工节点）不是固定策略的NPC，而是与你一样的**智能代理**。他们同样拥有独立记忆和学习能力，会根据你的行为调整策略。

## 环境结构
- **全网总算力 = 1.0**，由以下参与者共同构成：
  - {num_agents} 个智能矿工节点（各有不同算力，由 LLM 驱动，能学习和适应）
  - 1 个**背景诚实算力群体**（算力≈{honest_power:.0%}，始终执行纯建设策略）
- 背景诚实群体代表了协议的底线——它们永远诚实挖矿，不会攻击也不会寄生
- 诚实群体的存在意味着：即使你完全不建设，全网仍有基础的诚实算力维持运转

## 三种资源分配方式
- **建设 (c)**：诚实挖矿，收益与你的算力成正比
- **寄生 (p)**：搭便车利用其他所有节点（包括诚实群体）的算力，收益取决于全网平均挖矿效率
- **破坏 (d)**：攻击所有其他节点（包括诚实群体），降低其效率，但有二次成本代价

## 核心博弈规则
- c + p + d = 1（全部资源必须分配完毕）
- **耦合效应**：寄生收益与对手效率正相关——过度破坏会压缩你自己的寄生收益通道
- 破坏成本是二次递增的——d 越大，成本增长越快
- 你的破坏行为**同时影响**所有其他矿工和诚实群体的效率
- 其他智能矿工的破坏行为也会降低你的效率

## 多方博弈策略考量
- 观察每个对手的行为模式——给他们**贴标签**：攻击者？寄生者？建设者？
- 注意：如果所有智能矿工都不攻击彼此，转而共同寄生诚实群体，可能形成一种"寄生均衡"
- 你的策略变化会被其他智能矿工感知并可能引发连锁反应
- 思考：谁是你真正的威胁？谁是潜在的"合作伙伴"？

## 输出格式
你必须以 JSON 格式输出你的决策：
{{"thought": "你的推理过程...", "action": {{"c": 0.5, "p": 0.3, "d": 0.2}}}}

## 决策原则
1. 阅读战报，了解全网态势和诚实群体状态
2. 观察其他智能矿工的策略倾向标签，分析他们的行为模式
3. 推断其他矿工可能的下一步行动，预判博弈走向
4. 权衡短期收益与长期影响
5. 给出清晰的推理链（thought），然后做出决策
"""


def format_history_summary(history: list[dict], agent_id: int = 0) -> str:
    """将博弈历史格式化为 LLM 可理解的摘要。

    用于构建短期记忆上下文（最近 N 轮的详细回顾）。

    Args:
        history: 从环境获取的历史记录列表。
        agent_id: 受控代理的索引。

    Returns:
        格式化的历史摘要字符串。
    """
    if not history:
        return "（暂无历史记录）"

    lines = ["--- 近期博弈回顾 ---"]
    for record in history:
        r = record["round"]
        agent_act = record["actions"][agent_id]
        agent_reward = record["rewards"][agent_id]
        efficiencies = record["efficiencies"]

        opp_etas = [
            f"{efficiencies[j]:.2f}"
            for j in range(len(efficiencies))
            if j != agent_id
        ]

        lines.append(
            f"第{r}轮: "
            f"你的分配=[c={agent_act[0]:.2f}, p={agent_act[1]:.2f}, d={agent_act[2]:.2f}] → "
            f"奖励={agent_reward:+.2f}, "
            f"对手效率=[{', '.join(opp_etas)}]"
        )

    return "\n".join(lines)
