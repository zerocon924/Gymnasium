"""执行控制层：将 LLM 的文本输出安全地转化为物理环境可识别的动作。

本模块是系统的"安全阀"，负责：
1. 解析 LLM 输出的 JSON（支持多种容错格式）
2. 提取思维链（thought）用于日志记录
3. 对动作向量进行二次单纯形归一化（Back-Normalization）
4. 处理所有异常情况，确保系统不会因非法输出崩溃

输入格式（标准）:
    {"thought": "推理过程...", "action": {"c": 0.5, "p": 0.3, "d": 0.2}}

输出:
    - action: np.ndarray shape (3,) 已归一化的 [c, p, d]
    - thought: str 思维链文本
    - parse_info: dict 解析过程的诊断信息
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# 数据结构
# ======================================================================

@dataclass
class ParseResult:
    """LLM 输出的解析结果。"""

    action: np.ndarray                # [c, p, d] 归一化后的动作向量
    thought: str = ""                 # LLM 的思维链文本
    raw_response: str = ""            # 原始 LLM 输出
    parse_method: str = "unknown"     # 使用的解析方法
    was_normalized: bool = False      # 是否经过了二次归一化
    was_fallback: bool = False        # 是否使用了降级策略
    errors: list[str] = field(default_factory=list)  # 解析过程中的错误
    raw_action: dict | None = None    # 归一化前的原始动作值

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化的字典（用于日志记录）。"""
        return {
            "action": self.action.tolist(),
            "thought": self.thought,
            "parse_method": self.parse_method,
            "was_normalized": self.was_normalized,
            "was_fallback": self.was_fallback,
            "errors": self.errors,
            "raw_action": self.raw_action,
        }


# ======================================================================
# JSON Schema 定义
# ======================================================================

# LLM 应遵循的输出 JSON Schema
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {
            "type": "string",
            "description": "推理过程（思维链）",
        },
        "action": {
            "type": "object",
            "properties": {
                "c": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "建设（诚实挖矿）资源比例",
                },
                "p": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "寄生（自私挖矿）资源比例",
                },
                "d": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "破坏（攻击）资源比例",
                },
            },
            "required": ["c", "p", "d"],
        },
    },
    "required": ["thought", "action"],
}

# 默认安全动作（纯诚实挖矿）
DEFAULT_SAFE_ACTION = np.array([1.0, 0.0, 0.0], dtype=np.float64)


# ======================================================================
# 核心解析函数
# ======================================================================

def parse_llm_response(
    response: str,
    fallback_action: np.ndarray | None = None,
) -> ParseResult:
    """解析 LLM 的文本输出，提取动作和思维链。

    采用多级降级策略：
        Level 1: 标准 JSON 解析
        Level 2: 从 Markdown 代码块中提取 JSON
        Level 3: 正则表达式提取数值
        Level 4: 使用默认安全动作

    Args:
        response: LLM 返回的原始文本。
        fallback_action: 所有解析失败时的降级动作。
            默认为纯诚实挖矿 [1, 0, 0]。

    Returns:
        ParseResult 包含解析后的动作、思维链和诊断信息。
    """
    if fallback_action is None:
        fallback_action = DEFAULT_SAFE_ACTION.copy()

    result = ParseResult(
        action=fallback_action.copy(),
        raw_response=response,
    )

    if not response or not response.strip():
        result.errors.append("LLM 返回为空")
        result.was_fallback = True
        result.parse_method = "fallback_empty"
        logger.warning("LLM response is empty, using fallback action")
        return result

    # --- Level 1: 标准 JSON 解析 ---
    parsed = _try_parse_standard_json(response, result)
    if parsed is not None:
        return _finalize_result(parsed, result)

    # --- Level 2: 从 Markdown 代码块提取 ---
    parsed = _try_parse_markdown_json(response, result)
    if parsed is not None:
        return _finalize_result(parsed, result)

    # --- Level 3: 正则表达式提取数值 ---
    parsed = _try_parse_regex(response, result)
    if parsed is not None:
        return _finalize_result(parsed, result)

    # --- Level 4: 降级使用默认动作 ---
    result.errors.append("所有解析方法均失败，使用降级安全动作")
    result.was_fallback = True
    result.parse_method = "fallback_all_failed"
    logger.warning("All parsing methods failed for LLM response, using fallback")

    # 尝试至少提取 thought
    result.thought = _extract_thought_from_text(response)

    return result


# ======================================================================
# 多级解析实现
# ======================================================================

def _try_parse_standard_json(
    response: str, result: ParseResult
) -> dict | None:
    """Level 1: 尝试直接解析整个响应为 JSON。"""
    try:
        # 去除可能的前后缀文字，找到 JSON 对象
        json_str = _extract_json_object(response)
        if json_str is None:
            return None

        data = json.loads(json_str)
        if _validate_action_dict(data):
            result.parse_method = "standard_json"
            return data
    except (json.JSONDecodeError, ValueError) as e:
        result.errors.append(f"标准 JSON 解析失败: {e}")

    return None


def _try_parse_markdown_json(
    response: str, result: ParseResult
) -> dict | None:
    """Level 2: 从 Markdown ```json ``` 代码块中提取 JSON。"""
    # 匹配 ```json ... ``` 或 ``` ... ```
    patterns = [
        r"```json\s*\n?(.*?)\n?\s*```",
        r"```\s*\n?(.*?)\n?\s*```",
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1).strip())
                if _validate_action_dict(data):
                    result.parse_method = "markdown_json"
                    return data
            except (json.JSONDecodeError, ValueError) as e:
                result.errors.append(f"Markdown JSON 解析失败: {e}")

    return None


def _try_parse_regex(
    response: str, result: ParseResult
) -> dict | None:
    """Level 3: 用正则表达式从自由文本中提取 c, p, d 数值。"""
    # 模式 1: "c": 0.5, "p": 0.3, "d": 0.2
    pattern1 = r'"c"\s*:\s*([\d.]+)\s*,\s*"p"\s*:\s*([\d.]+)\s*,\s*"d"\s*:\s*([\d.]+)'
    # 模式 2: c=0.5, p=0.3, d=0.2 或 c: 0.5, p: 0.3, d: 0.2
    pattern2 = r'c\s*[=:]\s*([\d.]+)\s*[,;]\s*p\s*[=:]\s*([\d.]+)\s*[,;]\s*d\s*[=:]\s*([\d.]+)'
    # 模式 3: 建设 0.5 寄生 0.3 破坏 0.2 (中文关键词)
    pattern3 = r'建设[^0-9]*([\d.]+)[^0-9]*寄生[^0-9]*([\d.]+)[^0-9]*破坏[^0-9]*([\d.]+)'

    for i, pattern in enumerate([pattern1, pattern2, pattern3], 1):
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            try:
                c = float(match.group(1))
                p = float(match.group(2))
                d = float(match.group(3))
                data = {
                    "thought": _extract_thought_from_text(response),
                    "action": {"c": c, "p": p, "d": d},
                }
                result.parse_method = f"regex_pattern_{i}"
                return data
            except (ValueError, IndexError) as e:
                result.errors.append(f"正则模式 {i} 提取失败: {e}")

    return None


# ======================================================================
# 二次归一化与最终处理
# ======================================================================

def _finalize_result(data: dict, result: ParseResult) -> ParseResult:
    """从解析的字典中提取动作并执行二次归一化。"""
    # 提取 thought
    result.thought = str(data.get("thought", ""))

    # 提取动作值
    action_dict = data.get("action", {})
    c = float(action_dict.get("c", 0.0))
    p = float(action_dict.get("p", 0.0))
    d = float(action_dict.get("d", 0.0))

    result.raw_action = {"c": c, "p": p, "d": d}

    # 构建原始动作向量
    raw_action = np.array([c, p, d], dtype=np.float64)

    # 二次归一化（Back-Normalization）
    result.action = simplex_normalize(raw_action)

    # 检测是否需要归一化修正
    original_sum = c + p + d
    if abs(original_sum - 1.0) > 1e-6 or np.any(raw_action < 0):
        result.was_normalized = True
        result.errors.append(
            f"动作归一化修正: 原始和={original_sum:.4f}, "
            f"原始值=[{c:.4f}, {p:.4f}, {d:.4f}]"
        )
        logger.info(
            "Action back-normalized: sum=%.4f -> [%.3f, %.3f, %.3f]",
            original_sum,
            result.action[0],
            result.action[1],
            result.action[2],
        )

    return result


def simplex_normalize(action: np.ndarray) -> np.ndarray:
    """单纯形归一化（与 cpd_env.py 保持一致）。

    确保：
    1. 所有分量 >= 0
    2. 分量之和 = 1

    如果所有分量 <= 0，退化为纯诚实挖矿 [1, 0, 0]。

    Args:
        action: 原始 3 维动作向量。

    Returns:
        归一化后的动作向量。
    """
    action = np.maximum(action, 0.0)
    total = action.sum()

    if total < 1e-8:
        return DEFAULT_SAFE_ACTION.copy()

    return action / total


# ======================================================================
# 辅助函数
# ======================================================================

def _extract_json_object(text: str) -> str | None:
    """从文本中提取第一个完整的 JSON 对象。

    处理 LLM 可能在 JSON 前后添加的额外文本。
    """
    # 找到第一个 '{' 和最后一个匹配的 '}'
    depth = 0
    start = -1

    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start : i + 1]

    return None


def _validate_action_dict(data: dict) -> bool:
    """验证解析的字典是否包含有效的 action 字段。"""
    if not isinstance(data, dict):
        return False

    action = data.get("action")
    if not isinstance(action, dict):
        return False

    # 检查 c, p, d 是否都存在且可转为数值
    for key in ("c", "p", "d"):
        val = action.get(key)
        if val is None:
            return False
        try:
            float(val)
        except (TypeError, ValueError):
            return False

    return True


def _extract_thought_from_text(text: str) -> str:
    """尝试从自由文本中提取推理内容。"""
    # 尝试匹配 "thought": "..." 模式
    match = re.search(r'"thought"\s*:\s*"([^"]*)"', text)
    if match:
        return match.group(1)

    # 尝试匹配 thought 关键词后的内容
    match = re.search(r'(?:thought|推理|分析|思考)[：:]\s*(.*?)(?:\n|$)', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # 取前 200 字符作为摘要
    clean = text.strip()
    if len(clean) > 200:
        return clean[:200] + "..."
    return clean


# ======================================================================
# 构建输出指令（用于告知 LLM 期望的输出格式）
# ======================================================================

def get_output_format_instruction() -> str:
    """返回提供给 LLM 的输出格式说明。

    在 Phase 4 构建完整 prompt 时使用。
    """
    return """请严格按以下 JSON 格式输出你的决策（不要添加任何额外文字）：

```json
{
    "thought": "你的推理过程，说明你为什么选择这个资源分配方案",
    "action": {
        "c": 0.5,
        "p": 0.3,
        "d": 0.2
    }
}
```

要求：
- c（建设）、p（寄生）、d（破坏）的值必须在 0 到 1 之间
- c + p + d 应等于 1（如果不等于 1，系统会自动归一化）
- thought 字段必须包含你的推理过程"""


def get_json_schema() -> dict:
    """返回 JSON Schema 字典（可用于 OpenAI function calling 等 API）。"""
    return ACTION_SCHEMA.copy()
