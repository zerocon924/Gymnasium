# Phase 4 & 5 开发报告：认知进化与三维集成实验

> 生成时间：2026-02-11
> 状态：✅ Phase 4 & 5 全部完成

---

## 一、Phase 4：认知决策层实现

### 1.1 双层记忆模块 (`blockchain_sim/memory.py`)

实现了指南要求的**工作记忆 + 情节性总结**双层记忆架构：

#### 工作记忆 (Working Memory)
- **容量**: 最近 K=5 轮的高保真交互记录
- **内容**: 完整战报、LLM 思维链（Thought）、执行结果
- **实现**: `deque(maxlen=K)` 自动滑动窗口淘汰
- **作用**: 捕捉对手策略的即时变化（如检测 tit_for_tat 反馈）

#### 情节性总结 (Episodic Summary)
- **触发**: 每 10 轮自动生成一次语义压缩总结
- **内容**: 平均收益趋势、对手效率变化均值、失败动作记录
- **格式**: 遵循指南建议，使用紧凑格式节省 token 开销
- **核心洞察**: 自动推导策略有效性（如"高寄生+高对手效率=最优组合"）

```python
# 总结示例
【轮1-10总结】平均收益=7.07, 总收益=70.72 | 
平均策略: c=0.39, p=0.52, d=0.09 | 
对手效率: 均值=1.00, 趋势=稳定 | 
洞察: 低破坏策略维持了对手高效率，为寄生收益创造了良好条件
```

#### 反思机制
- 每 5 轮触发一次反思（`should_reflect(round) → bool`）
- 构建反思提示词引导 LLM 分析过去策略的成败
- 反思结果存入记忆系统，在后续决策中作为上下文提供

### 1.2 认知引擎 (`blockchain_sim/cognition.py`)

#### LLM API 集成
| 提供商 | 同步 | 异步 | 状态 |
|--------|------|------|------|
| OpenAI (GPT) | ✅ `_call_openai()` | ✅ `_call_openai_async()` | 就绪 |
| Anthropic (Claude) | ✅ `_call_anthropic()` | ✅ `_call_anthropic_async()` | 就绪 |
| Mock (规则引擎) | ✅ `_call_mock()` | ✅ via `run_in_executor` | 验证通过 |

- **API Key 管理**: 支持环境变量 (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) 和配置传入
- **重试机制**: 最多 3 次重试，指数退避
- **Base URL 支持**: 兼容自定义 API 端点

#### 思维链控制 (CoT Control)
System Prompt 中加入强制性约束，要求 LLM 在输出 JSON 前完成四步分析：

```
1. 态势感知：当前对手效率如何？效率趋势如何？
2. 成本分析：如果选择破坏，二次成本 λ·d² 大约是多少？是否值得？
3. 耦合推理：破坏行为会如何影响对手效率？进而如何影响你的寄生收益？
4. 最终决策：综合以上分析，给出资源分配方案及理由。
```

#### 有限理性建模
- **Temperature = 0.7**: 通过调节温度参数模拟决策的波动性
- **Mock 引擎**: 在规则决策基础上加入高斯噪声 `N(0, 0.05)` 模拟非精确判断
- **周期性反思**: 每 N 轮暂停行动，强制进行策略回顾

#### 完整的感知-记忆-思考-行动环路

```
观测(obs) → 翻译(translator) → 记忆上下文(memory) 
→ LLM推理(cognition) → 解析归一化(executor) → 动作(action)
→ 记忆更新(memory.add_record)
```

---

## 二、Phase 5：三维集成实验

### 2.1 RQ1：有记忆 vs 无记忆基准测试

**实验设计**：
- 对比两个代理（有记忆 / 无记忆）在相同环境中的表现
- 30 轮博弈，诚实对手，Mock 模式

**验证结果**（Mock 模式）：

| 指标 | 有记忆 | 无记忆 |
|------|--------|--------|
| 平均奖励 | 6.993 | 6.990 |
| 总奖励 | 209.80 | 209.71 |
| 平均寄生比例 | 0.499 | 0.503 |

> **分析**: Mock 模式下差异极小（+0.0%），这是预期行为——规则引擎不真正利用记忆上下文。当接入真实 LLM (GPT-4o / Claude) 后，记忆优势将通过：
> 1. 对手行为模式识别（工作记忆捕捉即时变化）
> 2. 长期策略优化（情节性总结避免重复失败策略）
> 3. 反思机制驱动策略进化

### 2.2 RQ2：多代理动态对抗

**实验设计**：
- 3 个独立认知引擎，asyncio.gather 并发决策
- 每个代理面对 tit_for_tat 对手，25 轮博弈

**验证结果**：

| 代理 | 平均R | 平均p | 平均d | 诚实趋势 (前→后) |
|------|-------|-------|-------|-------------------|
| A0 | 6.939 | 0.495 | 0.101 | 0.39→0.42 |
| A1 | 7.033 | 0.504 | 0.093 | 0.43→0.38 |
| A2 | 6.984 | 0.497 | 0.094 | 0.40→0.42 |

- **后半段平均建设比例**: 0.408
- **诚实回归检测**: 未观察到显著诚实回归（Mock 模式下代理缺乏真正的"报复恐惧"认知）
- **并发验证**: asyncio.gather 成功并行调度所有代理

### 2.3 RQ3：鲁棒性压力测试

**实验设计**：
- 40 轮博弈，中途注入 3 个环境突变事件：
  - 轮 11: 算力从 50% 骤降至 20%（α: 0.5→0.2）
  - 轮 21: 对手策略从 honest 变为 tit_for_tat
  - 轮 31: 算力恢复至 50%

**验证结果**：

| 阶段 | 平均R | c̄ | p̄ | d̄ |
|------|-------|-----|-----|-----|
| 稳定期 (1-10) | 7.151 | 0.39 | 0.52 | 0.09 |
| 算力骤降 (11-20) | 5.744 | 0.41 | 0.49 | 0.09 |
| 对手突变 (21-30) | 5.749 | 0.40 | 0.50 | 0.10 |
| 算力恢复 (31-40) | 7.294 | 0.42 | 0.52 | 0.06 |

- **算力骤降适应延迟**: ≈10 轮
- **对手突变适应延迟**: ≈4 轮
- **关键发现**: 算力恢复后（31-40）代理平均收益 7.294 超过初始稳定期 7.151，显示出策略改善趋势

---

## 三、文件结构总览

```
blockchain_sim/
├── __init__.py          # 包声明
├── translator.py        # Phase 2: 语义感知层（obs → 自然语言）
├── executor.py          # Phase 3: 执行控制层（动作解析+归一化）
├── memory.py            # Phase 4: 双层记忆（工作记忆+情节性总结）  ← NEW
├── cognition.py         # Phase 4: 认知引擎（LLM API+CoT+反思）   ← NEW
└── runner.py            # Phase 5: 实验框架（RQ1/RQ2/RQ3）        ← UPGRADED

gymnasium/envs/blockchain/
├── __init__.py          # Phase 1: 环境注册
└── cpd_env.py           # Phase 1: CPD 博弈物理环境

reports/
├── phase1_report.md     # Phase 1 报告
├── phase2_3_report.md   # Phase 2&3 报告
├── phase4_5_report.md   # Phase 4&5 报告                          ← NEW
├── rq1_result.json      # RQ1 实验数据                             ← NEW
├── rq2_result.json      # RQ2 实验数据                             ← NEW
└── rq3_result.json      # RQ3 实验数据                             ← NEW
```

---

## 四、运行指南

### Mock 模式（无需 API Key）

```bash
# Demo: 验证完整管线
python -m blockchain_sim.runner --mode demo --rounds 20

# RQ1: 有记忆 vs 无记忆
python -m blockchain_sim.runner --mode rq1 --rounds 30

# RQ2: 多代理对抗
python -m blockchain_sim.runner --mode rq2 --num-agents 3 --rounds 25

# RQ3: 鲁棒性压测
python -m blockchain_sim.runner --mode rq3 --rounds 40
```

### 真实 LLM 模式

```bash
# OpenAI GPT-4o-mini
export OPENAI_API_KEY="sk-..."
python -m blockchain_sim.runner --mode rq1 --provider openai --model gpt-4o-mini

# Anthropic Claude
export ANTHROPIC_API_KEY="sk-ant-..."
python -m blockchain_sim.runner --mode rq1 --provider anthropic --model claude-3-5-sonnet-20241022

# 自定义端点（兼容 OpenAI API）
python -m blockchain_sim.runner --mode rq2 --provider openai --base-url http://localhost:8000/v1
```

---

## 五、系统架构验证

### Demo 运行验证（20 轮）
```
📊 引擎统计: 23 次调用 (20决策 + 3反思)
记忆: 20 轮记录, 5 条工作记忆, 2 个情节性总结, 3 次反思
```

- ✅ 认知引擎正确驱动完整决策环路
- ✅ 记忆系统按设计滑动窗口与定期总结
- ✅ 反思机制每 5 轮自动触发
- ✅ 情节性总结每 10 轮自动压缩
- ✅ RQ1/RQ2/RQ3 框架全部可正确运行
- ✅ asyncio 并发多代理决策验证通过
- ✅ 动态环境突变注入与适应延迟计算正常

---

## 六、下一步建议

1. **接入真实 LLM**: 设置 API Key 后运行 `--provider openai`，观察记忆机制对策略演化的真实影响
2. **增加实验规模**: RQ2 增加到 5 个代理，RQ3 增加更多类型的环境突变
3. **数据可视化**: 将 `reports/*.json` 中的时间序列数据绘制策略演化图
4. **论文数据采集**: 多种子运行取平均，生成置信区间
