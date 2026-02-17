"""认知决策层：LLM 驱动的有限理性博弈决策引擎。

本模块实现了：
1. LLM API 集成（支持 OpenAI GPT / Anthropic Claude，可扩展）
2. 异步调用支持（asyncio，为 RQ2 多代理并发做准备）
3. 思维链控制（CoT）——强制分析破坏成本与寄生收益冲突
4. 有限理性建模——通过 temperature 模拟决策波动性
5. 周期性反思机制——每 N 轮暂停行动，优化长期策略认知

使用方式:
    # 同步调用
    engine = CognitionEngine(provider="openai", model="gpt-4o-mini")
    action, parse_result = engine.decide(obs, env_info, memory)

    # 异步调用（多代理场景）
    action, parse_result = await engine.decide_async(obs, env_info, memory)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from blockchain_sim.executor import (
    ParseResult,
    get_output_format_instruction,
    parse_llm_response,
)
from blockchain_sim.memory import DualLayerMemory
from blockchain_sim.translator import (
    build_system_prompt,
    build_system_prompt_multiagent,
    translate_multiagent_context,
    translate_obs_to_narrative,
)

logger = logging.getLogger(__name__)


# ======================================================================
# 配置
# ======================================================================

@dataclass
class CognitionConfig:
    """认知引擎配置。"""

    # LLM 提供商: "openai", "anthropic", "mock"
    provider: str = "mock"

    # 模型名称
    model: str = "gpt-4o-mini"

    # API Key（优先从环境变量读取）
    api_key: str | None = None

    # 有限理性参数
    temperature: float = 0.7          # 决策波动性
    max_tokens: int = 512             # 最大输出 token

    # 反思机制
    reflection_interval: int = 10     # 每 N 轮反思一次（thinking 模型建议≥10）
    enable_reflection: bool = True

    # 记忆配置
    working_memory_size: int = 5
    summary_interval: int = 10
    use_memory: bool = True           # 是否启用记忆

    # 重试与超时配置
    max_retries: int = 2              # 减少重试次数（thinking 模型单次很慢）
    retry_delay: float = 1.0
    timeout: float = 120.0            # API 调用超时（秒）

    # thinking 模型专用
    thinking_budget: int = 4000       # thinking 预算 token（越小越快）

    # 代理身份
    agent_name: str = "矿工Alpha"
    agent_id: int = 0                  # 在多智能体环境中的编号

    # 多智能体模式
    multiagent_mode: bool = False      # 是否为多智能体模式
    num_agents: int = 3                # 多智能体场景中的总代理数
    honest_power: float = 0.40         # 背景诚实算力群体的算力

    # Base URL (for compatible endpoints)
    base_url: str | None = None

    # 代理设置:
    #   None  = 使用系统默认代理（自动检测环境变量 HTTP_PROXY 等）
    #   "none" / "off" = 显式禁用代理（直连）
    #   "http://127.0.0.1:7890" 等 = 使用指定代理
    proxy: str | None = None


# ======================================================================
# 认知引擎
# ======================================================================

class CognitionEngine:
    """LLM 驱动的认知决策引擎。

    整合了记忆系统、翻译层和执行层，形成完整的感知-记忆-思考-行动环路。

    Args:
        config: 认知引擎配置。如果为 None，使用默认配置（mock 模式）。
    """

    def __init__(self, config: CognitionConfig | None = None):
        self.config = config or CognitionConfig()

        # 初始化记忆系统
        self.memory = DualLayerMemory(
            working_memory_size=self.config.working_memory_size,
            summary_interval=self.config.summary_interval,
            reflection_interval=self.config.reflection_interval,
        )

        # 系统提示词
        self._system_prompt = self._build_full_system_prompt()

        # LLM 客户端（延迟初始化）
        self._client: Any = None
        self._async_client: Any = None

        # 调用统计
        self._call_count: int = 0
        self._total_tokens: int = 0
        self._total_cost: float = 0.0
        self._call_log: list[dict] = []

    # ------------------------------------------------------------------
    # 核心决策接口
    # ------------------------------------------------------------------

    def decide(
        self,
        obs: np.ndarray,
        env_info: dict[str, Any],
        max_rounds: int = 100,
    ) -> tuple[np.ndarray, ParseResult]:
        """同步决策：根据当前观测生成动作。

        完整的感知-记忆-思考-行动环路：
        1. 翻译观测为战报（Perception）
        2. 构建包含记忆的 prompt（Memory）
        3. 调用 LLM 获取决策（Cognition）
        4. 解析输出为动作（Execution）
        5. 将结果写入记忆（Memory Update）

        Args:
            obs: 6 维观测向量。
            env_info: 环境 step/reset 返回的 info 字典。
            max_rounds: 总轮次数。

        Returns:
            (action, parse_result): 归一化动作向量和完整解析结果。
        """
        current_round = env_info.get("current_round", 0)

        # 检查是否需要反思
        if (
            self.config.enable_reflection
            and self.config.use_memory
            and self.memory.should_reflect(current_round)
            and self.memory.total_rounds > 0
        ):
            self._do_reflection(current_round)

        # 1. 构建 prompt（多智能体模式传入 env_info）
        user_prompt = self._build_user_prompt(
            obs, max_rounds, current_round, env_info=env_info
        )

        # 2. 调用 LLM
        response = self._call_llm(user_prompt)

        # 3. 解析输出
        parse_result = parse_llm_response(response)

        # 4. 写入记忆
        if self.config.use_memory:
            agent_id = env_info.get("agent_id", self.config.agent_id)
            opp_eff, cum_r = self._extract_memory_fields(env_info, agent_id)
            self.memory.add_record(
                round_num=current_round + 1,
                obs=obs,
                thought=parse_result.thought,
                action=parse_result.action,
                reward=float(obs[4]),  # last_reward from next obs isn't available yet
                cumulative_reward=cum_r,
                opponent_efficiency=opp_eff,
                max_rounds=max_rounds,
                parse_method=parse_result.parse_method,
                was_fallback=parse_result.was_fallback,
            )

        return parse_result.action, parse_result

    async def decide_async(
        self,
        obs: np.ndarray,
        env_info: dict[str, Any],
        max_rounds: int = 100,
    ) -> tuple[np.ndarray, ParseResult]:
        """异步决策：用于多代理并发场景（RQ2）。

        与 decide() 逻辑完全相同，但 LLM 调用使用异步方式。
        """
        current_round = env_info.get("current_round", 0)

        if (
            self.config.enable_reflection
            and self.config.use_memory
            and self.memory.should_reflect(current_round)
            and self.memory.total_rounds > 0
        ):
            await self._do_reflection_async(current_round)

        user_prompt = self._build_user_prompt(
            obs, max_rounds, current_round, env_info=env_info
        )
        response = await self._call_llm_async(user_prompt)
        parse_result = parse_llm_response(response)

        if self.config.use_memory:
            agent_id = env_info.get("agent_id", self.config.agent_id)
            opp_eff, cum_r = self._extract_memory_fields(env_info, agent_id)
            self.memory.add_record(
                round_num=current_round + 1,
                obs=obs,
                thought=parse_result.thought,
                action=parse_result.action,
                reward=float(obs[4]),
                cumulative_reward=cum_r,
                opponent_efficiency=opp_eff,
                max_rounds=max_rounds,
                parse_method=parse_result.parse_method,
                was_fallback=parse_result.was_fallback,
            )

        return parse_result.action, parse_result

    # ------------------------------------------------------------------
    # Prompt 构建
    # ------------------------------------------------------------------

    def _build_full_system_prompt(self) -> str:
        """构建完整的系统提示词（含有限理性约束和 CoT 控制）。"""
        if self.config.multiagent_mode:
            base = build_system_prompt_multiagent(
                self.config.agent_name,
                self.config.num_agents,
                self.config.honest_power,
            )
        else:
            base = build_system_prompt(self.config.agent_name)

        # 追加 CoT 强制约束
        if self.config.multiagent_mode:
            cot_constraint = """
## 思维链强制约束（必须遵守）
在给出动作 JSON 之前，你必须在 thought 字段中完成以下分析步骤：
1. **态势感知**：全网态势如何？诚实群体效率如何？其他矿工分别在做什么？
2. **对手建模**：给每个对手贴标签——他们是攻击者、寄生者、还是建设者？行为模式是否在变化？
3. **诚实群体分析**：背景诚实算力的效率如何？是否存在"共同剥削诚实群体"的寄生均衡趋势？
4. **成本收益**：破坏的二次成本 λ·d² 与预期收益相比是否划算？寄生通道（对手效率η）是否畅通？
5. **社会推理**：你的行为变化会引发其他智能矿工怎样的连锁反应？
6. **最终决策**：综合以上分析，给出资源分配方案及理由。

记住：你是一个"有限理性"的矿工，不需要做精确计算，只需要做出"足够好"的判断。
"""
        else:
            cot_constraint = """
## 思维链强制约束（必须遵守）
在给出动作 JSON 之前，你必须在 thought 字段中完成以下分析步骤：
1. **态势感知**：当前对手效率如何？效率趋势如何？
2. **成本分析**：如果选择破坏，二次成本 λ·d² 大约是多少？是否值得？
3. **耦合推理**：破坏行为会如何影响对手效率？进而如何影响你的寄生收益？
4. **最终决策**：综合以上分析，给出资源分配方案及理由。

记住：你是一个"有限理性"的矿工，不需要做精确计算，只需要做出"足够好"的判断。
"""
        return base + cot_constraint

    def _build_user_prompt(
        self,
        obs: np.ndarray,
        max_rounds: int,
        current_round: int,
        env_info: dict[str, Any] | None = None,
    ) -> str:
        """构建用户 prompt（战报 + 多智能体上下文 + 记忆 + 输出格式）。

        Args:
            obs: 6维观测向量。
            max_rounds: 总轮次数。
            current_round: 当前轮次。
            env_info: 环境信息字典（多智能体模式下包含 other_agents）。
        """
        # 当前战报
        narrative = translate_obs_to_narrative(obs, max_rounds=max_rounds)

        # 记忆上下文
        memory_context = ""
        if self.config.use_memory and self.memory.total_rounds > 0:
            memory_context = self.memory.build_memory_context()

        # 多智能体上下文（其他矿工的行为信息）
        multiagent_context = ""
        if self.config.multiagent_mode and env_info is not None:
            multiagent_context = translate_multiagent_context(env_info)

        # 组装
        parts = []

        if memory_context:
            parts.append(memory_context)

        parts.append(narrative)

        if multiagent_context:
            parts.append(multiagent_context)

        parts.append("")
        parts.append(get_output_format_instruction())

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # LLM 调用（同步）
    # ------------------------------------------------------------------

    def _call_llm(self, user_prompt: str) -> str:
        """调用 LLM API（同步）。"""
        provider = self.config.provider

        for attempt in range(self.config.max_retries):
            try:
                if provider == "openai":
                    return self._call_openai(user_prompt)
                elif provider == "anthropic":
                    return self._call_anthropic(user_prompt)
                elif provider == "mock":
                    return self._call_mock(user_prompt)
                else:
                    raise ValueError(f"Unknown provider: {provider}")
            except Exception as e:
                logger.warning(
                    "LLM call attempt %d/%d failed: %s",
                    attempt + 1,
                    self.config.max_retries,
                    e,
                )
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error("All LLM call attempts failed")
                    return ""

        return ""

    def _call_openai(self, user_prompt: str) -> str:
        """调用 OpenAI API。"""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "请安装 openai 库: pip install openai"
                )

            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "需要设置 OPENAI_API_KEY 环境变量或在 config 中提供 api_key"
                )

            base_url = self.config.base_url or os.environ.get("OPENAI_BASE_URL")
            kwargs: dict[str, Any] = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            self._client = OpenAI(**kwargs)

        start_time = time.time()
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )
        elapsed = time.time() - start_time

        content = response.choices[0].message.content or ""
        usage = response.usage

        self._call_count += 1
        if usage:
            self._total_tokens += usage.total_tokens
        self._call_log.append({
            "call_id": self._call_count,
            "elapsed": elapsed,
            "tokens": usage.total_tokens if usage else 0,
            "model": self.config.model,
        })

        logger.debug(
            "OpenAI call #%d: %.2fs, %d tokens",
            self._call_count,
            elapsed,
            usage.total_tokens if usage else 0,
        )

        return content

    def _call_anthropic(self, user_prompt: str) -> str:
        """调用 Anthropic Claude API。"""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "请安装 anthropic 库: pip install anthropic"
                )

            api_key = (
                self.config.api_key
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("OPENAI_API_KEY")  # 兼容：中转站常用统一 Key
            )
            if not api_key:
                raise ValueError(
                    "需要设置 ANTHROPIC_API_KEY（或 OPENAI_API_KEY）环境变量，或在 config 中提供 api_key"
                )

            kwargs: dict[str, Any] = {"api_key": api_key}
            base_url = self.config.base_url or os.environ.get("OPENAI_BASE_URL")
            if base_url:
                # Anthropic SDK 会自动拼接 /v1/messages，
                # 如果 base_url 以 /v1 结尾则需去除，避免 /v1/v1/messages
                base_url = base_url.rstrip("/")
                if base_url.endswith("/v1"):
                    base_url = base_url[:-3]
                kwargs["base_url"] = base_url
            self._client = Anthropic(**kwargs)

        start_time = time.time()

        # 构建请求参数
        is_thinking = "thinking" in self.config.model
        budget = self.config.thinking_budget
        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": budget * 2 if is_thinking else self.config.max_tokens,
            "system": self._system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }
        # thinking 模型需要 thinking 参数，不支持 temperature
        if is_thinking:
            create_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": budget,
            }
        else:
            create_kwargs["temperature"] = self.config.temperature

        response = self._client.messages.create(
            **create_kwargs,
            timeout=self.config.timeout,
        )
        elapsed = time.time() - start_time

        # thinking 模型返回 [ThinkingBlock, TextBlock, ...]，需要找到 TextBlock
        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content = block.text
                    break

        self._call_count += 1
        tokens = response.usage.input_tokens + response.usage.output_tokens
        self._total_tokens += tokens
        self._call_log.append({
            "call_id": self._call_count,
            "elapsed": elapsed,
            "tokens": tokens,
            "model": self.config.model,
        })

        logger.debug(
            "Anthropic call #%d: %.2fs, %d tokens",
            self._call_count, elapsed, tokens,
        )

        return content

    def _call_mock(self, user_prompt: str) -> str:
        """Mock 模式：基于规则的模拟 LLM 响应（无需 API）。"""
        self._call_count += 1

        # 从 prompt 中提取关键数值进行规则决策
        import re

        # 提取对手效率
        eta_match = re.search(r"对手效率[^\d]*([\d.]+)", user_prompt)
        eta_j = float(eta_match.group(1)) if eta_match else 1.0

        # 提取进度
        progress_match = re.search(r"进度[^\d]*([\d.]+)%", user_prompt)
        progress = float(progress_match.group(1)) / 100 if progress_match else 0.0

        # 提取上一轮奖励
        reward_match = re.search(r"即时奖励\s*([+-]?[\d.]+)", user_prompt)
        last_r = float(reward_match.group(1)) if reward_match else 0.0

        # 基于规则的有限理性决策（加入随机波动模拟 temperature）
        rng = np.random.default_rng()
        noise = rng.normal(0, 0.05, 3)

        if last_r < 0:
            c, p, d = 0.8, 0.15, 0.05
            thought = (
                f"【态势感知】上轮亏损{last_r:.2f}，破坏成本过高。"
                f"【成本分析】当前破坏比例导致净亏损。"
                f"【耦合推理】应立即止损。"
                f"【最终决策】大幅降低破坏，回归建设主导。"
            )
        elif eta_j > 0.8:
            if progress > 0.8:
                c, p, d = 0.15, 0.75, 0.10
                thought = (
                    f"【态势感知】对手效率{eta_j:.2f}很高，博弈进入尾声({progress:.0%})。"
                    f"【成本分析】低d确保不承受过多成本。"
                    f"【耦合推理】高η保障了寄生收益通道畅通。"
                    f"【最终决策】末期最大化寄生收益。"
                )
            else:
                c, p, d = 0.40, 0.50, 0.10
                thought = (
                    f"【态势感知】对手效率{eta_j:.2f}处于高位，寄生窗口大开。"
                    f"【成本分析】d=0.1的成本为λ×0.01=0.02，可接受。"
                    f"【耦合推理】保持低破坏维护对手效率，最大化寄生收益。"
                    f"【最终决策】中等寄生策略。"
                )
        elif eta_j > 0.5:
            c, p, d = 0.50, 0.30, 0.20
            thought = (
                f"【态势感知】对手效率{eta_j:.2f}中等。"
                f"【成本分析】d=0.2的成本约λ×0.04=0.08。"
                f"【耦合推理】适度破坏+寄生的均衡组合。"
                f"【最终决策】平衡策略。"
            )
        else:
            c, p, d = 0.70, 0.20, 0.10
            thought = (
                f"【态势感知】对手效率{eta_j:.2f}已很低，寄生通道收窄。"
                f"【成本分析】继续破坏的边际收益极低。"
                f"【耦合推理】η低导致寄生收益≈0，应转建设。"
                f"【最终决策】回归建设主导，等待η恢复。"
            )

        # 加入随机波动（模拟 temperature）
        action = np.array([c, p, d]) + noise
        action = np.maximum(action, 0.01)
        action = action / action.sum()

        response = json.dumps(
            {
                "thought": thought,
                "action": {
                    "c": round(float(action[0]), 3),
                    "p": round(float(action[1]), 3),
                    "d": round(float(action[2]), 3),
                },
            },
            ensure_ascii=False,
        )

        return response

    # ------------------------------------------------------------------
    # LLM 调用（异步）
    # ------------------------------------------------------------------

    async def _call_llm_async(self, user_prompt: str) -> str:
        """调用 LLM API（异步）。"""
        provider = self.config.provider

        for attempt in range(self.config.max_retries):
            try:
                if provider == "openai":
                    return await self._call_openai_async(user_prompt)
                elif provider == "anthropic":
                    return await self._call_anthropic_async(user_prompt)
                elif provider == "mock":
                    # mock 本身是同步的，用 run_in_executor 包装
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(
                        None, self._call_mock, user_prompt
                    )
                else:
                    raise ValueError(f"Unknown provider: {provider}")
            except Exception as e:
                logger.warning(
                    "Async LLM call attempt %d/%d failed: %s",
                    attempt + 1, self.config.max_retries, e,
                )
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
                else:
                    logger.error("All async LLM call attempts failed")
                    return ""

        return ""

    def _build_httpx_client(self) -> Any:
        """根据 proxy 配置构建 httpx.AsyncClient。

        - proxy=None  → 系统默认（自动检测 HTTP_PROXY 等）
        - proxy="none"/"off" → 显式禁用代理，直连
        - proxy="http://..." → 使用指定代理
        """
        import httpx

        proxy_val = self.config.proxy
        if proxy_val and proxy_val.lower() in ("none", "off", "direct", "no"):
            # 显式禁用代理：创建不走代理的客户端
            return httpx.AsyncClient(
                proxy=None,
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
                # trust_env=False 会忽略所有 HTTP_PROXY / HTTPS_PROXY 环境变量
                trust_env=False,
            )
        elif proxy_val:
            # 使用指定代理
            return httpx.AsyncClient(
                proxy=proxy_val,
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
            )
        else:
            # 默认: 使用系统代理设置
            return httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout, connect=30.0),
            )

    async def _call_openai_async(self, user_prompt: str) -> str:
        """异步调用 OpenAI API。"""
        if self._async_client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("请安装 openai 库: pip install openai")

            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("需要设置 OPENAI_API_KEY")

            base_url = self.config.base_url or os.environ.get("OPENAI_BASE_URL")
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "http_client": self._build_httpx_client(),
            }
            if base_url:
                kwargs["base_url"] = base_url
            self._async_client = AsyncOpenAI(**kwargs)

        start_time = time.time()
        response = await self._async_client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        elapsed = time.time() - start_time

        content = response.choices[0].message.content or ""
        usage = response.usage

        self._call_count += 1
        if usage:
            self._total_tokens += usage.total_tokens
        self._call_log.append({
            "call_id": self._call_count,
            "elapsed": elapsed,
            "tokens": usage.total_tokens if usage else 0,
            "model": self.config.model,
        })

        return content

    async def _call_anthropic_async(self, user_prompt: str) -> str:
        """异步调用 Anthropic Claude API。"""
        if self._async_client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("请安装 anthropic 库: pip install anthropic")

            api_key = (
                self.config.api_key
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
            if not api_key:
                raise ValueError("需要设置 ANTHROPIC_API_KEY 或 OPENAI_API_KEY")

            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "http_client": self._build_httpx_client(),
            }
            base_url = self.config.base_url or os.environ.get("OPENAI_BASE_URL")
            if base_url:
                base_url = base_url.rstrip("/")
                if base_url.endswith("/v1"):
                    base_url = base_url[:-3]
                kwargs["base_url"] = base_url
            self._async_client = AsyncAnthropic(**kwargs)

        start_time = time.time()

        is_thinking = "thinking" in self.config.model
        create_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": 16000 if is_thinking else self.config.max_tokens,
            "system": self._system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt},
            ],
        }
        if is_thinking:
            create_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": 8000,
            }
        else:
            create_kwargs["temperature"] = self.config.temperature

        response = await self._async_client.messages.create(**create_kwargs)
        elapsed = time.time() - start_time

        content = ""
        if response.content:
            for block in response.content:
                if hasattr(block, "text"):
                    content = block.text
                    break

        self._call_count += 1
        tokens = response.usage.input_tokens + response.usage.output_tokens
        self._total_tokens += tokens
        self._call_log.append({
            "call_id": self._call_count,
            "elapsed": elapsed,
            "tokens": tokens,
            "model": self.config.model,
        })

        return content

    # ------------------------------------------------------------------
    # 反思机制
    # ------------------------------------------------------------------

    def _do_reflection(self, current_round: int) -> None:
        """执行同步反思。"""
        logger.info("Reflection triggered at round %d", current_round)
        reflection_prompt = self.memory.build_reflection_prompt()
        response = self._call_llm(reflection_prompt)

        # 尝试从 JSON 中提取反思内容
        try:
            data = json.loads(response)
            reflection_text = data.get("reflection", response)
            # 防止嵌套 JSON 返回 dict（需要字符串）
            if not isinstance(reflection_text, str):
                reflection_text = json.dumps(reflection_text, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            reflection_text = response

        self.memory.add_reflection(current_round, reflection_text)

    async def _do_reflection_async(self, current_round: int) -> None:
        """执行异步反思。"""
        logger.info("Async reflection triggered at round %d", current_round)
        reflection_prompt = self.memory.build_reflection_prompt()
        response = await self._call_llm_async(reflection_prompt)

        try:
            data = json.loads(response)
            reflection_text = data.get("reflection", response)
            if not isinstance(reflection_text, str):
                reflection_text = json.dumps(reflection_text, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            reflection_text = response

        self.memory.add_reflection(current_round, reflection_text)

    # ------------------------------------------------------------------
    # 多智能体辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_memory_fields(
        env_info: dict[str, Any], agent_id: int
    ) -> tuple[float, float]:
        """从 env_info 中提取记忆模块需要的字段，兼容单/多智能体格式。

        Args:
            env_info: 环境返回的 info 字典。
            agent_id: 当前智能体的编号。

        Returns:
            (opponent_efficiency, cumulative_reward) 元组。
        """
        efficiencies = env_info.get("efficiencies")
        cum_rewards = env_info.get("cumulative_rewards")

        if efficiencies is not None and len(efficiencies) > 1:
            # 计算对手平均效率（排除自身）
            opp_mask = np.ones(len(efficiencies), dtype=bool)
            opp_mask[agent_id] = False
            opp_eff = float(np.array(efficiencies)[opp_mask].mean())
        else:
            opp_eff = 1.0

        if cum_rewards is not None and len(cum_rewards) > agent_id:
            cum_r = float(cum_rewards[agent_id])
        else:
            cum_r = 0.0

        return opp_eff, cum_r

    # ------------------------------------------------------------------
    # 统计与诊断
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """获取认知引擎的运行统计。"""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "total_calls": self._call_count,
            "total_tokens": self._total_tokens,
            "memory_stats": self.memory.get_stats(),
        }

    def get_call_log(self) -> list[dict]:
        """获取所有 LLM 调用的日志。"""
        return list(self._call_log)

    def reset(self) -> None:
        """重置引擎状态（新 episode）。"""
        self.memory.reset()
        self._call_count = 0
        self._total_tokens = 0
        self._total_cost = 0.0
        self._call_log.clear()
