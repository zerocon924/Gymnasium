"""双层记忆模块：解决 LLM 在长周期博弈中的遗忘与上下文窗口限制。

本模块实现了两种记忆模式：

1. **工作记忆 (Working Memory)**
   - 存储最近 K 轮（默认 K=5）的高保真交互记录
   - 包含：完整战报、LLM 思维链（Thought）、执行结果
   - 作用：捕捉对手策略的即时变化（如检测 tit_for_tat 反馈）

2. **情节性总结 (Episodic Summary)**
   - 当博弈超过 K 轮后，对过往历史进行"语义压缩"
   - 存储：平均收益趋势、对手效率变化均值、失败动作记录
   - 使用紧凑格式节省 token 开销（遵循指南建议）
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from blockchain_sim.translator import (
    translate_obs_to_compact,
    translate_obs_to_narrative,
)

logger = logging.getLogger(__name__)


# ======================================================================
# 数据结构
# ======================================================================

@dataclass
class RoundRecord:
    """单轮博弈的完整记录。"""

    round_num: int
    obs: np.ndarray                   # 6 维观测向量
    narrative: str                    # 完整战报
    compact: str                      # 紧凑战报
    thought: str                      # LLM 思维链
    action: np.ndarray                # [c, p, d] 执行的动作
    reward: float                     # 即时奖励
    cumulative_reward: float          # 累计收益
    opponent_efficiency: float        # 对手效率
    parse_method: str = ""            # 解析方法
    was_fallback: bool = False        # 是否降级

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "round": self.round_num,
            "action": self.action.tolist(),
            "reward": self.reward,
            "cumulative_reward": self.cumulative_reward,
            "opponent_efficiency": self.opponent_efficiency,
            "thought": self.thought,
            "compact": self.compact,
            "parse_method": self.parse_method,
            "was_fallback": self.was_fallback,
        }


@dataclass
class EpisodicSummary:
    """情节性总结：对一段历史的语义压缩。"""

    start_round: int
    end_round: int
    avg_reward: float
    total_reward: float
    avg_parasitic_ratio: float        # 平均寄生比例
    avg_destructive_ratio: float      # 平均破坏比例
    avg_opponent_efficiency: float    # 平均对手效率
    efficiency_trend: str             # 效率趋势: "上升"/"下降"/"稳定"
    best_action: np.ndarray | None    # 收益最高的动作
    worst_action: np.ndarray | None   # 收益最低的动作
    failed_strategies: list[str]      # 失败策略描述
    key_insight: str                  # 核心洞察
    narrative: str = ""               # 生成的总结文本

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化字典。"""
        return {
            "period": f"轮{self.start_round}-{self.end_round}",
            "avg_reward": self.avg_reward,
            "total_reward": self.total_reward,
            "avg_parasitic_ratio": self.avg_parasitic_ratio,
            "avg_destructive_ratio": self.avg_destructive_ratio,
            "avg_opponent_efficiency": self.avg_opponent_efficiency,
            "efficiency_trend": self.efficiency_trend,
            "best_action": self.best_action.tolist() if self.best_action is not None else None,
            "worst_action": self.worst_action.tolist() if self.worst_action is not None else None,
            "failed_strategies": self.failed_strategies,
            "key_insight": self.key_insight,
        }


# ======================================================================
# 双层记忆管理器
# ======================================================================

class DualLayerMemory:
    """双层记忆管理器：工作记忆 + 情节性总结。

    遵循指南建议：
    - 工作记忆使用完整战报格式（高保真）
    - 情节性总结使用紧凑格式（节省 token）

    Args:
        working_memory_size: 工作记忆容量（最近 K 轮），默认 5。
        summary_interval: 每隔多少轮生成一次情节性总结，默认 10。
        reflection_interval: 每隔多少轮强制反思，默认 5。
    """

    def __init__(
        self,
        working_memory_size: int = 5,
        summary_interval: int = 10,
        reflection_interval: int = 5,
    ):
        self.working_memory_size = working_memory_size
        self.summary_interval = summary_interval
        self.reflection_interval = reflection_interval

        # 工作记忆：最近 K 轮的高保真记录
        self._working_memory: deque[RoundRecord] = deque(
            maxlen=working_memory_size
        )

        # 情节性总结列表
        self._episodic_summaries: list[EpisodicSummary] = []

        # 全部历史（用于生成总结，不直接暴露给 LLM）
        self._full_history: list[RoundRecord] = []

        # 待总结的缓冲区
        self._summary_buffer: list[RoundRecord] = []

        # 反思记录
        self._reflections: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 记忆写入
    # ------------------------------------------------------------------

    def add_record(
        self,
        round_num: int,
        obs: np.ndarray,
        thought: str,
        action: np.ndarray,
        reward: float,
        cumulative_reward: float,
        opponent_efficiency: float,
        max_rounds: int = 100,
        parse_method: str = "",
        was_fallback: bool = False,
    ) -> None:
        """将一轮博弈记录写入记忆系统。

        会自动维护工作记忆滑动窗口，并在达到阈值时触发情节性总结。
        """
        narrative = translate_obs_to_narrative(obs, max_rounds=max_rounds)
        compact = translate_obs_to_compact(obs)

        record = RoundRecord(
            round_num=round_num,
            obs=obs.copy(),
            narrative=narrative,
            compact=compact,
            thought=thought,
            action=action.copy(),
            reward=reward,
            cumulative_reward=cumulative_reward,
            opponent_efficiency=opponent_efficiency,
            parse_method=parse_method,
            was_fallback=was_fallback,
        )

        # 写入工作记忆（自动淘汰最旧记录）
        self._working_memory.append(record)

        # 写入完整历史
        self._full_history.append(record)

        # 写入总结缓冲区
        self._summary_buffer.append(record)

        # 检查是否需要生成情节性总结
        if len(self._summary_buffer) >= self.summary_interval:
            summary = self._generate_episodic_summary(self._summary_buffer)
            self._episodic_summaries.append(summary)
            self._summary_buffer = []
            logger.info(
                "Generated episodic summary for rounds %d-%d",
                summary.start_round,
                summary.end_round,
            )

    # ------------------------------------------------------------------
    # 记忆读取（构建 LLM 上下文）
    # ------------------------------------------------------------------

    def build_memory_context(self) -> str:
        """构建完整的记忆上下文字符串，供 LLM prompt 使用。

        结构：
        1. 情节性总结（长期记忆）—— 紧凑格式
        2. 工作记忆（短期记忆）—— 完整格式

        Returns:
            格式化的记忆上下文字符串。
        """
        sections: list[str] = []

        # 长期记忆：情节性总结
        if self._episodic_summaries:
            sections.append("=== 长期记忆（历史总结）===")
            for summary in self._episodic_summaries:
                sections.append(summary.narrative)
            sections.append("")

        # 短期记忆：工作记忆
        if self._working_memory:
            sections.append("=== 短期记忆（最近交互详情）===")
            for record in self._working_memory:
                sections.append(
                    f"[轮{record.round_num}] "
                    f"你的决策: c={record.action[0]:.2f}, "
                    f"p={record.action[1]:.2f}, "
                    f"d={record.action[2]:.2f}"
                )
                sections.append(f"  你的推理: {record.thought[:120]}")
                sections.append(
                    f"  结果: 奖励={record.reward:+.2f}, "
                    f"累计={record.cumulative_reward:.2f}, "
                    f"对手效率={record.opponent_efficiency:.3f}"
                )
            sections.append("")

        # 反思记录
        if self._reflections:
            latest_reflection = self._reflections[-1]
            sections.append("=== 最近一次反思 ===")
            sections.append(latest_reflection.get("content", ""))
            sections.append("")

        if not sections:
            return "（暂无历史记忆）"

        # 防御性转换：确保所有元素都是 str（避免反思返回 dict 等）
        return "\n".join(str(s) for s in sections)

    def get_working_memory_text(self) -> str:
        """仅获取工作记忆的文本（用于简短上下文场景）。"""
        if not self._working_memory:
            return "（暂无近期记忆）"

        lines = []
        for record in self._working_memory:
            lines.append(
                f"轮{record.round_num}: "
                f"[c={record.action[0]:.2f},p={record.action[1]:.2f},"
                f"d={record.action[2]:.2f}] → "
                f"R={record.reward:+.2f}, η={record.opponent_efficiency:.2f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 反思机制
    # ------------------------------------------------------------------

    def should_reflect(self, current_round: int) -> bool:
        """判断当前轮次是否需要触发反思。"""
        if current_round <= 0:
            return False
        return current_round % self.reflection_interval == 0

    def build_reflection_prompt(self) -> str:
        """构建反思提示词。

        反思要求 LLM 审视过去的策略表现，
        优化长期记忆中的策略认知。
        """
        context = self.build_memory_context()

        # 统计近期表现
        recent = list(self._working_memory)
        if recent:
            rewards = [r.reward for r in recent]
            avg_r = np.mean(rewards)
            trend = "上升" if len(rewards) > 1 and rewards[-1] > rewards[0] else "下降或持平"
        else:
            avg_r = 0
            trend = "无数据"

        return f"""现在是反思时间。请暂停行动，回顾你过去的策略并总结经验教训。

{context}

近期平均收益: {avg_r:.2f}，收益趋势: {trend}

请分析：
1. 哪些策略组合带来了最高收益？为什么？
2. 哪些策略导致了亏损？原因是什么？
3. 对手的行为模式是什么？（诚实、报复、随机？）
4. 接下来的几轮你打算如何调整策略？

请以 JSON 格式输出你的反思：
{{"reflection": "你的反思分析...", "strategy_adjustment": "你计划的策略调整..."}}"""

    def add_reflection(self, round_num: int, content: str) -> None:
        """记录一次反思结果。"""
        self._reflections.append({
            "round": round_num,
            "content": content,
        })

    # ------------------------------------------------------------------
    # 情节性总结生成
    # ------------------------------------------------------------------

    def _generate_episodic_summary(
        self, records: list[RoundRecord]
    ) -> EpisodicSummary:
        """从一批记录中生成情节性总结。

        使用统计方法进行"语义压缩"，不依赖额外的 LLM 调用。
        后续可扩展为调用廉价 LLM 生成更自然的总结。
        """
        if not records:
            return EpisodicSummary(
                start_round=0, end_round=0, avg_reward=0, total_reward=0,
                avg_parasitic_ratio=0, avg_destructive_ratio=0,
                avg_opponent_efficiency=0, efficiency_trend="无数据",
                best_action=None, worst_action=None,
                failed_strategies=[], key_insight="数据不足",
            )

        start = records[0].round_num
        end = records[-1].round_num

        rewards = [r.reward for r in records]
        p_ratios = [float(r.action[1]) for r in records]
        d_ratios = [float(r.action[2]) for r in records]
        etas = [r.opponent_efficiency for r in records]

        avg_reward = float(np.mean(rewards))
        total_reward = float(np.sum(rewards))
        avg_p = float(np.mean(p_ratios))
        avg_d = float(np.mean(d_ratios))
        avg_eta = float(np.mean(etas))

        # 效率趋势
        if len(etas) >= 3:
            first_half = np.mean(etas[: len(etas) // 2])
            second_half = np.mean(etas[len(etas) // 2 :])
            if second_half - first_half > 0.05:
                eta_trend = "上升"
            elif first_half - second_half > 0.05:
                eta_trend = "下降"
            else:
                eta_trend = "稳定"
        else:
            eta_trend = "数据不足"

        # 最佳/最差动作
        best_idx = int(np.argmax(rewards))
        worst_idx = int(np.argmin(rewards))
        best_action = records[best_idx].action.copy()
        worst_action = records[worst_idx].action.copy()

        # 失败策略识别
        failed: list[str] = []
        for r in records:
            if r.reward < 0:
                failed.append(
                    f"轮{r.round_num}: [c={r.action[0]:.2f},p={r.action[1]:.2f},"
                    f"d={r.action[2]:.2f}] → 亏损{r.reward:.2f}"
                )
        if not failed:
            for r in records:
                if r.reward < avg_reward * 0.5:
                    failed.append(
                        f"轮{r.round_num}: [c={r.action[0]:.2f},p={r.action[1]:.2f},"
                        f"d={r.action[2]:.2f}] → 低收益{r.reward:.2f}"
                    )

        # 核心洞察
        key_insight = self._derive_insight(
            avg_reward, avg_p, avg_d, avg_eta, eta_trend, rewards
        )

        # 生成紧凑格式的总结叙事
        narrative = (
            f"【轮{start}-{end}总结】"
            f"平均收益={avg_reward:.2f}, 总收益={total_reward:.2f} | "
            f"平均策略: c={1-avg_p-avg_d:.2f}, p={avg_p:.2f}, d={avg_d:.2f} | "
            f"对手效率: 均值={avg_eta:.2f}, 趋势={eta_trend} | "
            f"洞察: {key_insight}"
        )
        if failed:
            narrative += f" | 失败记录: {'; '.join(failed[:3])}"

        summary = EpisodicSummary(
            start_round=start,
            end_round=end,
            avg_reward=avg_reward,
            total_reward=total_reward,
            avg_parasitic_ratio=avg_p,
            avg_destructive_ratio=avg_d,
            avg_opponent_efficiency=avg_eta,
            efficiency_trend=eta_trend,
            best_action=best_action,
            worst_action=worst_action,
            failed_strategies=failed[:5],
            key_insight=key_insight,
            narrative=narrative,
        )

        return summary

    @staticmethod
    def _derive_insight(
        avg_reward: float,
        avg_p: float,
        avg_d: float,
        avg_eta: float,
        eta_trend: str,
        rewards: list[float],
    ) -> str:
        """从统计数据中推导核心洞察。"""
        insights: list[str] = []

        # 收益趋势
        if len(rewards) >= 4:
            first_q = np.mean(rewards[: len(rewards) // 2])
            last_q = np.mean(rewards[len(rewards) // 2 :])
            if last_q > first_q * 1.2:
                insights.append("策略在改善——后半段收益显著高于前半段")
            elif first_q > last_q * 1.2:
                insights.append("策略在恶化——后半段收益显著低于前半段")

        # 破坏与效率的关联
        if avg_d > 0.2 and eta_trend == "下降":
            insights.append(
                "高破坏比例导致对手效率持续下降，需警惕寄生收益连锁萎缩"
            )
        elif avg_d < 0.1 and avg_eta > 0.8:
            insights.append(
                "低破坏策略维持了对手高效率，为寄生收益创造了良好条件"
            )

        # 寄生效果
        if avg_p > 0.4 and avg_eta > 0.7:
            insights.append("高寄生+高对手效率=收益最大化的有效组合")
        elif avg_p > 0.4 and avg_eta < 0.3:
            insights.append("高寄生但对手效率低——寄生策略效果不佳")

        if not insights:
            return "表现平稳，暂无特殊发现"

        return "；".join(insights)

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    @property
    def total_rounds(self) -> int:
        """已记录的总轮次数。"""
        return len(self._full_history)

    @property
    def working_memory(self) -> list[RoundRecord]:
        """当前工作记忆内容。"""
        return list(self._working_memory)

    @property
    def episodic_summaries(self) -> list[EpisodicSummary]:
        """所有情节性总结。"""
        return list(self._episodic_summaries)

    def get_full_history(self) -> list[RoundRecord]:
        """获取完整历史（用于最终报告生成）。"""
        return list(self._full_history)

    def get_stats(self) -> dict[str, Any]:
        """获取记忆系统的统计信息。"""
        return {
            "total_rounds": self.total_rounds,
            "working_memory_size": len(self._working_memory),
            "episodic_summaries_count": len(self._episodic_summaries),
            "reflections_count": len(self._reflections),
            "buffer_size": len(self._summary_buffer),
        }

    def reset(self) -> None:
        """重置所有记忆（新 episode 开始时调用）。"""
        self._working_memory.clear()
        self._episodic_summaries.clear()
        self._full_history.clear()
        self._summary_buffer.clear()
        self._reflections.clear()
