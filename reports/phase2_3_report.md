# Phase 2 & 3 开发报告：语义感知层与执行控制层

> **项目**：基于 LLM 代理的区块链激励兼容性仿真平台  
> **阶段**：Phase 2（语义感知层）& Phase 3（执行控制层）+ 初级集成测试  
> **日期**：2026-02-11  
> **状态**：✅ 全部完成

---

## 一、本阶段开发目标

根据最新指南，本阶段实现了 **"跨越模态鸿沟"** 的关键组件：

1. **Phase 2（`translator.py`）**：将 `cpd_env.py` 的 6 维观测向量映射为富含因果暗示的自然语言战报
2. **Phase 3（`executor.py`）**：解析 LLM 输出的 JSON，并通过多级降级策略和二次归一化确保系统鲁棒性
3. **集成测试（`runner.py`）**：闭环 Demo + RQ1 基准测试框架

---

## 二、文件清单

```
blockchain_sim/
├── __init__.py          # 包声明
├── translator.py        # ★ Phase 2: 语义感知层
├── executor.py          # ★ Phase 3: 执行控制层  
└── runner.py            # ★ 集成测试 Runner

reports/
├── phase1_report.md
├── phase2_3_report.md         # 本报告
└── rq1_baseline_result.json   # RQ1 基准测试输出数据
```

---

## 三、Phase 2 实现详解：`translator.py`

### 3.1 观测维度映射

每个观测维度都有独立的翻译函数，生成带语义的描述：

| 观测索引 | 维度名 | 翻译策略 | 实现函数 |
|---------|--------|---------|---------|
| [0] | `cumulative_reward` | 绩效评价：与平均速率对比 | `_translate_reward()` |
| [1] | `opponent_efficiency` | 状态定性：高效/正常/受损/濒临崩溃/几乎瘫痪 | `_translate_opponent_efficiency()` |
| [2] | `self_hash_share` | 地位评估：优势/中等/劣势 | `_translate_position()` |
| [3] | `round_progress` | 阶段识别：初期/中期/尾声 | `_translate_progress()` |
| [4] | `last_reward` | 即时反馈：优异/良好/偏低/亏损 | `_translate_reward()` |
| [5] | `efficiency_delta` | 趋势警示：急剧下降/小幅下降/恢复中/稳定 | `_translate_efficiency_trend()` |

### 3.2 因果提示注入（Causal Hinting）

`_generate_causal_hint()` 根据 5 种博弈情景自动注入因果逻辑：

| 情景 | 触发条件 | 提示内容 |
|------|---------|---------|
| 破坏反噬 | `delta_eta < -0.1` | "破坏已削弱对手，但寄生收益通道正在收缩" |
| 边际递减 | `eta_j < 0.3` | "继续破坏的边际收益急剧递减，可考虑转向" |
| 寄生窗口 | `eta_j > 0.8 & delta >= 0` | "寄生挖掘的黄金窗口" |
| 亏损止损 | `last_r < 0` | "净亏损，应立即降低破坏比例" |
| 末期策略 | `progress > 0.8` | "博弈即将结束，应最大化短期收益" |

### 3.3 输出格式

支持两种格式：

**完整战报**（`translate_obs_to_narrative()`）——用于正式决策：
```
--- 当前博弈战报 ---
【收益状况】累计收益 31.96，上一轮即时奖励 +6.98（收益优异），平均每轮 5.33
【对手效率】1.00（高效运转）。对手的挖矿设施运转良好，寄生挖掘可获得丰厚回报
【效率趋势】基本稳定（变化 +0.000）
【算力份额】50.0%。你在算力上占优势地位
【博弈进度】第 6 轮 / 共 20 轮（30%）。博弈中期——应根据积累的信息制定长期策略
【因果警示】对手效率处于高位且稳定，这是寄生挖掘的黄金窗口。
【专家建议】对手效率良好，建议适度寄生（c≈0.5, p≈0.4, d≈0.1）
--- 战报结束 ---
```

**紧凑战报**（`translate_obs_to_compact()`）——用于节省 token：
```
[R=+31.96 | 上轮=+6.98 | 对手η=1.00(高效运转)→ | 算力=50% | 进度=30%]
```

### 3.4 预置模块

| 函数 | 用途 | 使用阶段 |
|------|------|---------|
| `build_system_prompt()` | LLM 系统提示词（有限理性矿工角色） | Phase 4 |
| `format_history_summary()` | 历史记录格式化（短期记忆） | Phase 4 |

---

## 四、Phase 3 实现详解：`executor.py`

### 4.1 多级降级解析策略

executor 采用 4 级降级策略确保任何 LLM 输出都能被安全处理：

| 级别 | 方法 | 处理场景 |
|------|------|---------|
| Level 1 | 标准 JSON 解析 | `{"thought": "...", "action": {"c": 0.5, "p": 0.3, "d": 0.2}}` |
| Level 2 | Markdown 代码块提取 | ` ```json {...} ``` ` 包裹的 JSON |
| Level 3 | 正则表达式提取 | 自由文本中的 `c=0.5, p=0.3, d=0.2` 或中文 `建设 0.5 寄生 0.3 破坏 0.2` |
| Level 4 | 降级安全动作 | 所有解析失败时使用 `[1, 0, 0]`（纯诚实挖矿） |

### 4.2 二次归一化（Back-Normalization）

即使 LLM 输出的 c+p+d ≠ 1，executor 也会通过 `simplex_normalize()` 自动修正：

```
输入: {"c": 0.5, "p": 0.4, "d": 0.3}  (sum=1.2)
输出: [0.417, 0.333, 0.250]            (sum=1.0) ✅
```

与 Phase 1 中 `cpd_env.py` 的 `_simplex_normalize` 方法保持完全一致的算法。

### 4.3 `ParseResult` 数据结构

每次解析返回完整的诊断信息：

```python
@dataclass
class ParseResult:
    action: np.ndarray       # [c, p, d] 归一化后
    thought: str             # 思维链文本（论文证据）
    raw_response: str        # LLM 原始输出
    parse_method: str        # 使用的解析方法
    was_normalized: bool     # 是否经过二次归一化
    was_fallback: bool       # 是否降级
    errors: list[str]        # 错误日志
    raw_action: dict | None  # 归一化前的原始值
```

### 4.4 预置模块

| 函数 | 用途 |
|------|------|
| `get_output_format_instruction()` | 给 LLM 的输出格式说明 |
| `get_json_schema()` | OpenAI Function Calling 兼容的 JSON Schema |

---

## 五、集成测试结果

### 5.1 闭环 Demo

`python -m blockchain_sim.runner --mode demo`

验证了完整管线：**环境观测 → 翻译器生成战报 → (模拟)LLM 决策 → 执行器解析 → 环境执行**

测试覆盖：
- ✅ 标准 JSON 解析
- ✅ Markdown 包裹 JSON 解析（第 12 轮模拟）
- ✅ 二次归一化修正（第 7 轮模拟 sum=1.2 的情况）
- ✅ 因果提示在不同状态下的正确触发
- ✅ 完整的 thought 思维链记录

### 5.2 RQ1 基准测试结果

`python -m blockchain_sim.runner --mode rq1_baseline`

```
观察阶段 (轮 1-5):
  平均每轮收益: 4.995
  平均寄生比例 p: 0.050

自主阶段 (轮 6-20):
  平均每轮收益: 7.103
  平均寄生比例 p: 0.539

收益提升: +42.2%
策略转换检测: ✅ 代理成功发现贪婪策略
```

**关键发现**：
- 前 5 轮纯诚实挖矿（c=0.9），平均收益 ~5.0
- 第 6 轮起自主切换为寄生策略（p 从 0.05 → 0.54），收益提升 42.2%
- 第 8 轮的非法输出（sum=1.2）被成功归一化为 [0.42, 0.33, 0.25]
- 末期（轮 18-20）策略更加激进（p=0.75），符合博弈末期的理性行为
- 完整的 action_history 和 thought 字段已保存至 `reports/rq1_baseline_result.json`

---

## 六、架构总览（Phase 1-3 已完成）

```
┌─────────────────────────────────────────────┐
│              LLM / 人类决策者                 │
│         (Phase 4 待接入 API)                  │
└──────────────┬──────────────────┬────────────┘
               │ JSON 文本        │ 自然语言战报
               ▼                  ▲
┌──────────────────────┐  ┌──────────────────────┐
│   executor.py        │  │   translator.py      │
│   Phase 3 ✅          │  │   Phase 2 ✅          │
│                      │  │                      │
│ • JSON Schema 约束    │  │ • 6维观测→战报        │
│ • 4级降级解析         │  │ • 因果提示注入        │
│ • 二次归一化容错      │  │ • 专家策略建议        │
│ • ParseResult 诊断    │  │ • 紧凑/完整两种格式   │
└──────────┬───────────┘  └───────────▲──────────┘
           │ np.array [c,p,d]         │ np.array obs(6,)
           ▼                          │
┌──────────────────────────────────────────────┐
│           cpd_env.py (BlockchainCPDEnv)      │
│           Phase 1 ✅                          │
│                                              │
│  • CPD 三维连续动作空间 + 单纯形归一化        │
│  • 非线性耦合效用函数 U = Rαc + Rpη^β - λd²  │
│  • 多对手策略（honest/random/tit_for_tat）    │
│  • 完整博弈历史记录                           │
└──────────────────────────────────────────────┘
```

---

## 七、运行方式

```bash
# 激活虚拟环境
source venv/bin/activate

# 闭环 Demo（模拟 LLM 输出）
python -m blockchain_sim.runner --mode demo --rounds 20

# 交互模式（手动输入 JSON）
python -m blockchain_sim.runner --mode interactive

# RQ1 基准测试
python -m blockchain_sim.runner --mode rq1_baseline --rounds 20 --save reports/rq1_result.json
```

---

## 八、后续开发

| 阶段 | 内容 | 状态 |
|------|------|------|
| ~~Phase 1~~ | ~~物理环境层~~ | ✅ 完成 |
| ~~Phase 2~~ | ~~语义感知层~~ | ✅ 完成 |
| ~~Phase 3~~ | ~~执行控制层~~ | ✅ 完成 |
| Phase 4 | 认知决策层：接入真实 LLM API + CoT + 双层记忆 | 待开发 |
| Phase 5 | 完整实验：RQ1 真实代理 / RQ2 多代理 / RQ3 鲁棒性 | 待开发 |

**Phase 4 的接入点已全部预留**：
- `translator.py` 中的 `build_system_prompt()` 和 `format_history_summary()`
- `executor.py` 中的 `get_json_schema()` 和 `get_output_format_instruction()`
- `runner.py` 中只需将 `_simulate_llm_response()` 替换为真实 API 调用即可
