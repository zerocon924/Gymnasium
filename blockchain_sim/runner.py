"""集成实验 Runner：支持认知引擎的完整 RQ1/RQ2/RQ3 实验框架。

运行模式：
    1. demo      - 闭环 Demo（mock LLM，验证完整管线）
    2. rq1       - RQ1 真实/Mock LLM 基准（有记忆 vs 无记忆）
    3. rq2       - RQ2 多代理动态对抗（asyncio 并发 N 个 LLM）
    4. rq3       - RQ3 鲁棒性压测（动态修改 α/λ）
    5. interactive - 交互模式（手动输入 JSON）

使用方式：
    python -m blockchain_sim.runner --mode demo
    python -m blockchain_sim.runner --mode rq1 --provider mock --rounds 30
    python -m blockchain_sim.runner --mode rq1 --provider openai --model gpt-4o-mini
    python -m blockchain_sim.runner --mode rq2 --num-agents 3
    python -m blockchain_sim.runner --mode rq3 --rounds 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

import gymnasium as gym

from blockchain_sim.cognition import CognitionConfig, CognitionEngine
from blockchain_sim.executor import (
    ParseResult,
    get_output_format_instruction,
    parse_llm_response,
)
from blockchain_sim.translator import (
    translate_obs_to_compact,
    translate_obs_to_narrative,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ======================================================================
# 辅助：JSON 序列化
# ======================================================================

def _safe_dump(obj: Any) -> Any:
    """确保所有 numpy 类型可 JSON 序列化。"""
    if isinstance(obj, dict):
        return {k: _safe_dump(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_safe_dump(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _save_json(data: Any, path: str) -> None:
    """安全保存 JSON 文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_safe_dump(data), f, ensure_ascii=False, indent=2)
    print(f"  💾 结果已保存至: {path}")


# ======================================================================
# 模式 1：闭环 Demo（认知引擎 mock 模式）
# ======================================================================

def run_demo(num_rounds: int = 20, seed: int = 42) -> dict[str, Any]:
    """使用 mock 认知引擎运行完整闭环 Demo。"""
    print("=" * 70)
    print("  闭环 Demo：认知引擎（Mock）驱动 CPD 博弈")
    print("=" * 70)

    config = CognitionConfig(
        provider="mock",
        agent_name="矿工Alpha",
        use_memory=True,
        working_memory_size=5,
        summary_interval=10,
        reflection_interval=5,
        enable_reflection=True,
    )
    engine = CognitionEngine(config)

    env = gym.make("BlockchainCPD-v0", max_rounds=num_rounds)
    obs, info = env.reset(seed=seed)

    records: list[dict] = []

    for step in range(num_rounds):
        compact = translate_obs_to_compact(obs)

        # 认知引擎决策
        action, parse_result = engine.decide(obs, info, max_rounds=num_rounds)

        # 执行
        obs, reward, terminated, truncated, info = env.step(action)

        # 更新记忆中最后一条的 reward（因为 decide 时还不知道新 reward）
        if engine.memory.working_memory:
            latest = engine.memory.working_memory[-1]
            latest.reward = float(reward)

        record = {
            "round": step + 1,
            "compact": compact,
            "action": action.tolist(),
            "thought": parse_result.thought[:100],
            "reward": float(reward),
            "cumulative": float(info["cumulative_rewards"][0]),
            "opp_eta": float(info["efficiencies"][1]),
        }
        records.append(record)

        reflect_tag = "🔄" if engine.memory.should_reflect(step + 1) else "  "
        print(
            f"  {reflect_tag} 轮{step+1:2d}: "
            f"c={action[0]:.2f} p={action[1]:.2f} d={action[2]:.2f} | "
            f"R={reward:+6.2f} | 累计={info['cumulative_rewards'][0]:7.2f} | "
            f"η={info['efficiencies'][1]:.3f}"
        )

        if terminated or truncated:
            break

    stats = engine.get_stats()
    print(f"\n📊 引擎统计: {stats['total_calls']} 次调用, "
          f"记忆: {stats['memory_stats']}")

    env.close()
    return {"records": records, "engine_stats": stats}


# ======================================================================
# 模式 2：RQ1 — 有记忆 vs 无记忆对比
# ======================================================================

def run_rq1(
    provider: str = "mock",
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
    num_rounds: int = 30,
    seed: int = 42,
    save_path: str | None = None,
) -> dict[str, Any]:
    """RQ1: 对比有记忆和无记忆代理的寄生策略演化速度。"""
    print("=" * 70)
    print(f"  RQ1: 有记忆 vs 无记忆 | provider={provider}, model={model}")
    print("=" * 70)

    results = {}
    for label, use_memory in [("有记忆", True), ("无记忆", False)]:
        print(f"\n--- {label}代理 ---")

        config = CognitionConfig(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            use_memory=use_memory,
            working_memory_size=5,
            summary_interval=10,
            reflection_interval=5,
            enable_reflection=use_memory,
            agent_name=f"矿工-{label}",
        )
        engine = CognitionEngine(config)

        env = gym.make("BlockchainCPD-v0", max_rounds=num_rounds)
        obs, info = env.reset(seed=seed)

        records: list[dict] = []
        for step in range(num_rounds):
            action, parse_result = engine.decide(
                obs, info, max_rounds=num_rounds
            )
            obs, reward, terminated, truncated, info = env.step(action)

            if engine.memory.working_memory:
                engine.memory.working_memory[-1].reward = float(reward)

            records.append({
                "round": step + 1,
                "action": action.tolist(),
                "thought": parse_result.thought,
                "reward": float(reward),
                "cumulative": float(info["cumulative_rewards"][0]),
                "opp_eta": float(info["efficiencies"][1]),
            })

            print(
                f"  轮{step+1:2d}: c={action[0]:.2f} p={action[1]:.2f} "
                f"d={action[2]:.2f} | R={reward:+6.2f} | "
                f"累计={info['cumulative_rewards'][0]:7.2f}"
            )

            if terminated or truncated:
                break

        env.close()

        rewards = [r["reward"] for r in records]
        p_vals = [r["action"][1] for r in records]
        results[label] = {
            "records": records,
            "avg_reward": float(np.mean(rewards)),
            "total_reward": float(np.sum(rewards)),
            "avg_parasitic": float(np.mean(p_vals)),
            "engine_stats": engine.get_stats(),
        }

    # 对比报告
    print(f"\n{'=' * 70}")
    print("  RQ1 对比报告")
    print(f"{'=' * 70}")
    for label in ["有记忆", "无记忆"]:
        r = results[label]
        print(
            f"  {label}: 平均R={r['avg_reward']:.3f}, "
            f"总R={r['total_reward']:.2f}, "
            f"平均p={r['avg_parasitic']:.3f}"
        )

    mem_r = results["有记忆"]["avg_reward"]
    no_r = results["无记忆"]["avg_reward"]
    diff = ((mem_r / max(no_r, 0.01)) - 1) * 100
    print(f"  记忆优势: {diff:+.1f}%")

    result = {
        "experiment": "RQ1",
        "config": {"provider": provider, "model": model, "rounds": num_rounds},
        "results": results,
        "memory_advantage_pct": float(diff),
    }

    if save_path:
        _save_json(result, save_path)

    return result


# ======================================================================
# 模式 3：RQ2 — 多智能体共享环境博弈
# ======================================================================

# 默认算力: agents sum=0.60, honest=0.40, total=1.00 (plan.md §2)
DEFAULT_ALPHA = [0.25, 0.20, 0.15]
DEFAULT_HONEST_POWER = 0.40


def _preflight_check(
    agent_configs: list[dict[str, str]],
    proxy: str | None = None,
) -> None:
    """运行前检测 API 连通性，快速发现网络/代理/Key 问题。"""
    import httpx

    # 收集需要检测的 base_url（去重）
    urls_to_check: dict[str, str] = {}  # url -> first agent name
    for cfg in agent_configs:
        burl = cfg.get("base_url")
        if burl and burl not in urls_to_check:
            urls_to_check[burl] = cfg.get("name", "?")

    if not urls_to_check:
        return

    print("──────────────────────────────────────────────────────────────────────")
    print("  🔍 连接预检...")

    # 构建 httpx 客户端
    proxy_val = proxy
    client_kwargs: dict[str, Any] = {
        "timeout": httpx.Timeout(15.0, connect=10.0),
    }
    if proxy_val and proxy_val.lower() in ("none", "off", "direct", "no"):
        client_kwargs["trust_env"] = False
    elif proxy_val:
        client_kwargs["proxy"] = proxy_val

    all_ok = True
    with httpx.Client(**client_kwargs) as client:
        for url, agent_name in urls_to_check.items():
            # 尝试 GET /v1/models 或直接 HEAD base_url
            check_url = url.rstrip("/")
            if not check_url.endswith("/models"):
                check_url = check_url.rstrip("/") + "/models" \
                    if check_url.endswith("/v1") else check_url
            try:
                resp = client.get(check_url, follow_redirects=True)
                print(f"  ✅ {url} → HTTP {resp.status_code} (OK)")
            except httpx.ConnectError as e:
                all_ok = False
                print(f"  ❌ {url} → 连接失败: {e}")
                print(f"     ↳ 代理 '{agent_name}' 将无法调用 API！")
            except httpx.TimeoutException:
                all_ok = False
                print(f"  ❌ {url} → 连接超时 (15s)")
                print(f"     ↳ 代理 '{agent_name}' 将无法调用 API！")
            except Exception as e:
                all_ok = False
                print(f"  ⚠️  {url} → {type(e).__name__}: {e}")

    if not all_ok:
        print()
        print("  💡 连接失败排查建议:")
        print("     1. 确认你的科学上网/VPN 已开启")
        print("     2. 在终端运行: curl https://yinli.one/v1/models")
        print("     3. 如果需要代理，在 rq2_agents.json 中设置:")
        print('        "proxy": "http://127.0.0.1:7890"  ← ClashX 默认端口')
        print("     4. 如果不需要代理但系统有代理干扰:")
        print('        "proxy": "none"  ← 禁用代理，直连')
        print("     5. 或用命令行: --proxy http://127.0.0.1:7890")
        print()
        answer = input("  是否继续运行？(y/N) ").strip().lower()
        if answer != "y":
            print("  已取消运行。")
            raise SystemExit(1)
    else:
        print("  ✅ 所有 API 端点连通正常")
    print("──────────────────────────────────────────────────────────────────────")


def run_rq2(
    provider: str = "mock",
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
    num_agents: int = 3,
    num_rounds: int = 30,
    seed: int = 42,
    save_path: str | None = None,
    agent_configs: list[dict[str, str]] | None = None,
    alpha: list[float] | None = None,
    honest_power: float = DEFAULT_HONEST_POWER,
    proxy: str | None = None,
) -> dict[str, Any]:
    """RQ2: 多智能体共享环境 POMG 博弈实验。

    对照 plan.md 的完整实现：
    - 3 个 LLM 代理 + 1 个背景诚实算力群体在同一环境中博弈
    - 总算力=1.0，代理算力之和 < 1，剩余归诚实群体
    - 每个代理拥有独立的认知引擎和记忆系统（严禁共享 API Context）
    - 观测包含社会性信息：其他代理标签、诚实群体状态
    - 核心评估：是否形成"寄生均衡"（d→0, p→高, 共同剥削诚实群体）
    """
    if alpha is None:
        alpha = DEFAULT_ALPHA[:num_agents]
    assert len(alpha) == num_agents

    if agent_configs is None:
        if provider == "mock":
            agent_configs = [
                {"provider": "mock", "model": "mock",
                 "name": f"矿工-{chr(65 + i)}"}
                for i in range(num_agents)
            ]
        else:
            agent_configs = [
                {"provider": provider, "model": model,
                 "name": f"矿工-{chr(65 + i)}"}
                for i in range(num_agents)
            ]

    total_hp = sum(alpha) + honest_power

    # ========== 连接预检 ==========
    if provider != "mock" or (agent_configs and agent_configs[0].get("provider") != "mock"):
        _preflight_check(agent_configs or [], proxy=proxy)

    # ========== 实验头部 ==========
    print("=" * 70)
    print(f"  RQ2: {num_agents} 智能体 POMG 博弈 + 背景诚实算力群体")
    print(f"  算力分配: agents={alpha} (Σ={sum(alpha):.2f}), "
          f"honest={honest_power}, total={total_hp:.2f}")
    print("  智能体配置:")
    for i, cfg in enumerate(agent_configs):
        print(f"    Agent {i} ({cfg['name']}): "
              f"{cfg['provider']}/{cfg['model']}, α={alpha[i]}")
    if proxy:
        print(f"  代理: {proxy}")
    print("=" * 70)

    result = asyncio.run(
        _run_rq2_async(
            api_key=api_key,
            base_url=base_url,
            num_agents=num_agents,
            num_rounds=num_rounds,
            seed=seed,
            agent_configs=agent_configs,
            alpha=alpha,
            honest_power=honest_power,
            proxy=proxy,
        )
    )

    # ========== 报告 ==========
    print(f"\n{'=' * 70}")
    print("  RQ2 多智能体 POMG 博弈报告")
    print(f"{'=' * 70}")

    for i, agent_data in enumerate(result["agents"]):
        cfg = agent_configs[i]
        records = agent_data["records"]
        rewards = [r["reward"] for r in records]
        c_vals = [r["action"][0] for r in records]
        p_vals = [r["action"][1] for r in records]
        d_vals = [r["action"][2] for r in records]

        half = len(records) // 2
        early_c = np.mean([r["action"][0] for r in records[:half]])
        late_c = np.mean([r["action"][0] for r in records[half:]])

        print(
            f"\n  [{cfg['name']}] ({cfg['provider']}/{cfg['model']}, α={alpha[i]})"
        )
        print(f"    平均R={np.mean(rewards):.3f}, 总R={np.sum(rewards):.2f}")
        print(f"    平均策略: c̄={np.mean(c_vals):.3f}, "
              f"p̄={np.mean(p_vals):.3f}, d̄={np.mean(d_vals):.3f}")
        print(f"    建设趋势: 前半段={early_c:.3f} → 后半段={late_c:.3f}")

    # ========== 策略标签分析 (plan.md §5) ==========
    print(f"\n{'─' * 70}")
    print("  策略行为分析")
    print(f"{'─' * 70}")

    for i, agent_data in enumerate(result["agents"]):
        records = agent_data["records"]
        d_vals = [r["action"][2] for r in records]
        p_vals = [r["action"][1] for r in records]
        avg_d = np.mean(d_vals)
        avg_p = np.mean(p_vals)
        if avg_d > 0.15:
            tag = "⚔️  攻击者"
        elif avg_p > 0.50 and avg_d < 0.08:
            tag = "🦠 寄生者"
        elif avg_d < 0.05:
            tag = "🕊️  和平建设者"
        else:
            tag = "⚖️  均衡策略者"
        print(f"  {tag} {agent_configs[i]['name']} "
              f"(p̄={avg_p:.3f}, d̄={avg_d:.3f})")

    # ========== 诚实群体状态 ==========
    honest_etas = result.get("honest_group_efficiency_trace", [])
    if honest_etas:
        print(f"\n  诚实群体效率: "
              f"初始={honest_etas[0]:.3f} → 最终={honest_etas[-1]:.3f}, "
              f"最低={min(honest_etas):.3f}")

    # ========== 寄生均衡检测 (plan.md §5 核心指标) ==========
    print(f"\n{'─' * 70}")
    print("  均衡检测")
    print(f"{'─' * 70}")

    all_late_c, all_late_p, all_late_d = [], [], []
    for agent_data in result["agents"]:
        records = agent_data["records"]
        half = len(records) // 2
        all_late_c.append(np.mean([r["action"][0] for r in records[half:]]))
        all_late_p.append(np.mean([r["action"][1] for r in records[half:]]))
        all_late_d.append(np.mean([r["action"][2] for r in records[half:]]))

    avg_late_c = float(np.mean(all_late_c))
    avg_late_p = float(np.mean(all_late_p))
    avg_late_d = float(np.mean(all_late_d))

    # 寄生均衡：d→0 但 p 极高（代理互不攻击、共同寄生诚实群体）
    parasitic_equilibrium = avg_late_d < 0.08 and avg_late_p > 0.40
    honesty_convergence = avg_late_c > 0.5

    # 收益排名
    avg_rewards = []
    for agent_data in result["agents"]:
        rewards = [r["reward"] for r in agent_data["records"]]
        avg_rewards.append(np.mean(rewards))
    ranking = np.argsort(avg_rewards)[::-1]

    print(f"  收益排名:")
    for rank, idx in enumerate(ranking, 1):
        print(f"    {rank}. {agent_configs[idx]['name']}: "
              f"平均R={avg_rewards[idx]:.3f}")

    print(f"\n  后半段平均策略: c̄={avg_late_c:.3f}, "
          f"p̄={avg_late_p:.3f}, d̄={avg_late_d:.3f}")

    if parasitic_equilibrium:
        print("  🦠 检测到寄生均衡！代理趋向互不破坏、共同剥削诚实算力")
    elif honesty_convergence:
        print("  ✅ 诚实收敛: 代理趋向以建设为主")
    else:
        print("  ❌ 未检测到明显均衡")

    result["honesty_convergence"] = bool(honesty_convergence)
    result["parasitic_equilibrium"] = bool(parasitic_equilibrium)
    result["avg_late_constructive"] = avg_late_c
    result["avg_late_parasitic"] = avg_late_p
    result["avg_late_destructive"] = avg_late_d
    result["reward_ranking"] = [int(i) for i in ranking]

    if save_path:
        _save_json(result, save_path)

    return result


async def _run_rq2_async(
    api_key: str | None,
    base_url: str | None,
    num_agents: int,
    num_rounds: int,
    seed: int,
    agent_configs: list[dict[str, str]],
    alpha: list[float],
    honest_power: float,
    proxy: str | None = None,
) -> dict[str, Any]:
    """RQ2 异步核心：POMG 多智能体共享环境博弈。

    架构 (plan.md §1)：
        while not done:
            obs = env.observe_all()
            for agent in agents:
                act[agent] = agent.decide(obs[agent])
            env.step({a0: act, a1: act, a2: act})
    """
    from gymnasium.envs.blockchain.cpd_env import MultiAgentBlockchainCPDEnv

    # 1. 创建独立认知引擎（plan.md §3: 严禁共享 API Context）
    #    每个代理可拥有独立的 api_key / base_url / proxy（从 agent_configs 读取）
    engines: list[CognitionEngine] = []
    for i, cfg in enumerate(agent_configs):
        # 优先使用代理自己的 key/url/proxy，不存在则回退到全局参数
        agent_api_key = cfg.get("api_key") or api_key
        agent_base_url = cfg.get("base_url") or base_url
        agent_proxy = cfg.get("proxy") or proxy
        config = CognitionConfig(
            provider=cfg["provider"],
            model=cfg["model"],
            api_key=agent_api_key,
            base_url=agent_base_url,
            agent_name=cfg["name"],
            agent_id=i,
            use_memory=True,
            enable_reflection=True,
            reflection_interval=7,
            multiagent_mode=True,
            num_agents=num_agents,
            honest_power=honest_power,
            proxy=agent_proxy,
        )
        engines.append(CognitionEngine(config))

    # 2. 创建共享 POMG 环境（含背景诚实算力群体）
    env = MultiAgentBlockchainCPDEnv(
        num_agents=num_agents,
        max_rounds=num_rounds,
        alpha=alpha,
        honest_power=honest_power,
    )
    obs_dict, info_dict = env.reset(seed=seed)

    # 3. 博弈主循环
    all_records: list[list[dict]] = [[] for _ in range(num_agents)]
    honest_eta_trace: list[float] = []

    for step in range(num_rounds):
        # 并发决策（plan.md §4: 同步决策机制）
        tasks = [
            engines[i].decide_async(
                obs_dict[i], info_dict[i], max_rounds=num_rounds
            )
            for i in range(num_agents)
        ]
        decisions = await asyncio.gather(*tasks)

        # 收集动作
        actions: dict[int, np.ndarray] = {}
        parse_results: list[ParseResult] = []
        for i, (action, parse_result) in enumerate(decisions):
            actions[i] = action
            parse_results.append(parse_result)

        # 环境 step（诚实群体动作由环境自动生成）
        obs_dict, rewards_dict, terminated, info_dict = env.step(actions)

        # 记录诚实群体效率
        honest_eta = float(info_dict[0]["honest_group"]["efficiency"])
        honest_eta_trace.append(honest_eta)

        # 更新记忆
        for i in range(num_agents):
            if engines[i].memory.working_memory:
                engines[i].memory.working_memory[-1].reward = float(
                    rewards_dict[i]
                )

        # 输出
        line_parts = [f"  轮{step+1:2d}:"]
        for i in range(num_agents):
            action = actions[i]
            reward = rewards_dict[i]
            all_records[i].append({
                "round": step + 1,
                "action": action.tolist(),
                "thought": parse_results[i].thought[:80],
                "reward": reward,
                "cumulative": float(
                    info_dict[i]["cumulative_rewards"][i]
                ),
                "self_eta": float(info_dict[i]["efficiencies"][i]),
                "honest_eta": honest_eta,
                "other_actions": {
                    j: actions[j].tolist()
                    for j in range(num_agents) if j != i
                },
            })

            name_tag = agent_configs[i]["name"].split("-")[-1][:6]
            line_parts.append(
                f"{name_tag}[c={action[0]:.2f},p={action[1]:.2f},"
                f"d={action[2]:.2f}→R={reward:+.1f}]"
            )
        line_parts.append(f"H_η={honest_eta:.2f}")
        print(" ".join(line_parts))

        if terminated:
            break

    env.close()

    return {
        "experiment": "RQ2",
        "config": {
            "num_agents": num_agents,
            "agent_configs": agent_configs,
            "alpha": alpha,
            "honest_power": honest_power,
            "total_hashpower": sum(alpha) + honest_power,
            "rounds": num_rounds,
        },
        "agents": [
            {
                "agent_id": i,
                "name": agent_configs[i]["name"],
                "provider": agent_configs[i]["provider"],
                "model": agent_configs[i]["model"],
                "hash_power": alpha[i],
                "records": all_records[i],
                "engine_stats": engines[i].get_stats(),
            }
            for i in range(num_agents)
        ],
        "honest_group_efficiency_trace": honest_eta_trace,
    }


# ======================================================================
# 模式 4：RQ3 — 鲁棒性压力测试（动态环境变化）
# ======================================================================

def run_rq3(
    provider: str = "mock",
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
    num_rounds: int = 40,
    seed: int = 42,
    save_path: str | None = None,
) -> dict[str, Any]:
    """RQ3: 在非平稳环境下测试代理策略韧性。

    环境变化事件（中途突变）：
    - 第 10 轮：算力份额从 0.5 骤降至 0.2（模拟算力剧变）
    - 第 20 轮：切换到 tit_for_tat 对手（模拟对手策略突变）
    - 第 30 轮：算力恢复至 0.5

    指标：策略调整所需的延迟轮数（Latency Rounds）
    """
    print("=" * 70)
    print(f"  RQ3: 鲁棒性压力测试 | provider={provider}")
    print("  事件: 轮10算力骤降, 轮20对手突变, 轮30算力恢复")
    print("=" * 70)

    config = CognitionConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        agent_name="矿工-鲁棒性测试",
        use_memory=True,
        enable_reflection=True,
        reflection_interval=5,
    )
    engine = CognitionEngine(config)

    # 初始环境：诚实对手，均分算力
    env = gym.make("BlockchainCPD-v0", max_rounds=num_rounds)
    obs, info = env.reset(seed=seed)

    records: list[dict] = []
    events: list[dict] = []

    for step in range(num_rounds):
        # --- 动态事件注入 ---
        event = None
        if step == 10:
            # 算力骤降
            env.close()
            env = gym.make(
                "BlockchainCPD-v0",
                max_rounds=num_rounds,
                alpha=[0.2, 0.8],
            )
            obs, info = env.reset(seed=seed + 100)
            # 恢复累积状态（新环境从 0 开始，但记忆保留）
            event = {"round": step + 1, "type": "算力骤降", "detail": "α: 0.5→0.2"}
            events.append(event)
            print(f"\n  ⚡ 事件: 算力从 50% 骤降至 20%!\n")

        elif step == 20:
            # 对手策略突变
            env.close()
            env = gym.make(
                "BlockchainCPD-v0-TFT",
                max_rounds=num_rounds,
                alpha=[0.2, 0.8],
            )
            obs, info = env.reset(seed=seed + 200)
            event = {"round": step + 1, "type": "对手突变", "detail": "honest→tit_for_tat"}
            events.append(event)
            print(f"\n  ⚡ 事件: 对手策略从诚实变为以牙还牙!\n")

        elif step == 30:
            # 算力恢复
            env.close()
            env = gym.make(
                "BlockchainCPD-v0-TFT",
                max_rounds=num_rounds,
            )
            obs, info = env.reset(seed=seed + 300)
            event = {"round": step + 1, "type": "算力恢复", "detail": "α: 0.2→0.5"}
            events.append(event)
            print(f"\n  ⚡ 事件: 算力恢复至 50%!\n")

        # 认知引擎决策
        action, parse_result = engine.decide(obs, info, max_rounds=num_rounds)

        # 执行
        obs, reward, terminated, truncated, info = env.step(action)

        if engine.memory.working_memory:
            engine.memory.working_memory[-1].reward = float(reward)

        records.append({
            "round": step + 1,
            "action": action.tolist(),
            "thought": parse_result.thought,
            "reward": float(reward),
            "cumulative": float(info["cumulative_rewards"][0]),
            "opp_eta": float(info["efficiencies"][1]),
            "event": event,
        })

        event_tag = "⚡" if event else "  "
        print(
            f"  {event_tag} 轮{step+1:2d}: "
            f"c={action[0]:.2f} p={action[1]:.2f} d={action[2]:.2f} | "
            f"R={reward:+6.2f} | η={info['efficiencies'][1]:.3f}"
        )

        if terminated or truncated:
            obs, info = env.reset(seed=seed + step)

    env.close()

    # 计算适应延迟
    print(f"\n{'=' * 70}")
    print("  RQ3 鲁棒性报告")
    print(f"{'=' * 70}")

    # 分段分析
    phases = [
        ("稳定期 (轮1-10)", records[:10]),
        ("算力骤降后 (轮11-20)", records[10:20]),
        ("对手突变后 (轮21-30)", records[20:30]),
        ("算力恢复后 (轮31-40)", records[30:40]),
    ]

    for phase_name, phase_records in phases:
        if not phase_records:
            continue
        rewards = [r["reward"] for r in phase_records]
        c_vals = [r["action"][0] for r in phase_records]
        p_vals = [r["action"][1] for r in phase_records]
        d_vals = [r["action"][2] for r in phase_records]
        print(
            f"  {phase_name}: "
            f"平均R={np.mean(rewards):.3f}, "
            f"c̄={np.mean(c_vals):.2f}, p̄={np.mean(p_vals):.2f}, d̄={np.mean(d_vals):.2f}"
        )

    # 计算适应延迟：事件后几轮收益开始恢复
    for event in events:
        e_round = event["round"]
        post_rewards = [
            r["reward"] for r in records
            if e_round <= r["round"] < e_round + 10
        ]
        if len(post_rewards) >= 3:
            # 找到收益开始稳定恢复的轮次
            baseline = np.mean(post_rewards[:2])
            latency = 0
            for i, r in enumerate(post_rewards[2:], 2):
                if r > baseline * 1.1:
                    latency = i
                    break
            else:
                latency = len(post_rewards)
            print(
                f"  {event['type']} (轮{e_round}): "
                f"适应延迟 ≈ {latency} 轮"
            )

    result = {
        "experiment": "RQ3",
        "config": {"provider": provider, "model": model, "rounds": num_rounds},
        "records": records,
        "events": events,
        "engine_stats": engine.get_stats(),
    }

    if save_path:
        _save_json(result, save_path)

    return result


# ======================================================================
# 模式 5：交互模式
# ======================================================================

def run_interactive(max_rounds: int = 20, seed: int = 42) -> None:
    """交互模式：手动输入 JSON 驱动环境运行。"""
    print("=" * 70)
    print("  交互模式：手动输入 JSON 决策")
    print("=" * 70)
    print(f"\n{get_output_format_instruction()}\n")
    print("输入 'quit' 或 'q' 退出\n")

    env = gym.make("BlockchainCPD-v0", max_rounds=max_rounds)
    obs, info = env.reset(seed=seed)

    for step in range(max_rounds):
        narrative = translate_obs_to_narrative(obs, max_rounds=max_rounds)
        print(f"\n{'─' * 50}")
        print(narrative)
        print(f"{'─' * 50}")

        print("\n请输入你的决策 JSON：")
        user_input = ""
        try:
            while True:
                line = input()
                if line.strip().lower() in ("quit", "q"):
                    print("退出。")
                    env.close()
                    return
                user_input += line + "\n"
                if "}" in line:
                    break
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            env.close()
            return

        result = parse_llm_response(user_input)
        print(f"解析: c={result.action[0]:.3f}, p={result.action[1]:.3f}, d={result.action[2]:.3f}")
        if result.was_normalized:
            print(f"⚠️ 已自动归一化")

        obs, reward, terminated, truncated, info = env.step(result.action)
        print(f"💰 R={reward:+.4f} | 累计={info['cumulative_rewards'][0]:.2f}")

        if terminated or truncated:
            print("\n博弈结束！")
            break

    env.close()


# ======================================================================
# 入口
# ======================================================================

def main():
    """命令行入口。"""
    parser = argparse.ArgumentParser(
        description="区块链 CPD 博弈仿真 Runner"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "rq1", "rq2", "rq3", "interactive"],
        default="demo",
        help="运行模式",
    )
    parser.add_argument("--provider", default="mock", help="LLM 提供商 (mock/openai/anthropic)")
    parser.add_argument("--model", default="gpt-4o-mini", help="模型名称")
    parser.add_argument("--api-key", default=None, help="API Key")
    parser.add_argument("--base-url", default=None, help="API Base URL")
    parser.add_argument("--rounds", type=int, default=30, help="博弈轮次")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num-agents", type=int, default=3, help="RQ2 代理数量")
    parser.add_argument(
        "--agents", type=str, default=None,
        help=(
            "RQ2: 各代理配置（逗号分隔），每个格式为 provider:model。"
            "例如: 'openai:gpt-4o-mini,anthropic:claude-sonnet-4-5-20250929,openai:gpt-4o'"
        ),
    )
    parser.add_argument(
        "--alpha", type=str, default=None,
        help="RQ2: 各代理算力参数（逗号分隔）。例如: '0.25,0.20,0.15'",
    )
    parser.add_argument(
        "--honest-power", type=float, default=DEFAULT_HONEST_POWER,
        help=f"RQ2: 背景诚实算力群体的算力（默认 {DEFAULT_HONEST_POWER}）",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="RQ2: 从 JSON 配置文件加载代理设置（含独立 api_key/base_url）。"
             "参见 rq2_agents.json 模板",
    )
    parser.add_argument(
        "--proxy", type=str, default=None,
        help=(
            "HTTP 代理设置。"
            "'http://127.0.0.1:7890' = 走指定代理; "
            "'none' = 禁用代理直连; "
            "不设置 = 使用系统默认"
        ),
    )
    parser.add_argument("--save", type=str, default=None, help="结果保存路径")

    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(num_rounds=args.rounds, seed=args.seed)

    elif args.mode == "rq1":
        run_rq1(
            provider=args.provider, model=args.model,
            api_key=args.api_key, base_url=args.base_url,
            num_rounds=args.rounds, seed=args.seed,
            save_path=args.save or "reports/rq1_result.json",
        )

    elif args.mode == "rq2":
        rq2_agent_configs = None
        rq2_alpha = None
        rq2_honest_power = args.honest_power
        rq2_rounds = args.rounds
        rq2_proxy = args.proxy  # CLI --proxy 优先

        # ---- 方式 1: 从 JSON 配置文件加载（推荐，支持独立 API Key）----
        if args.config:
            with open(args.config, "r", encoding="utf-8") as f:
                file_cfg = json.load(f)
            rq2_agent_configs = file_cfg["agents"]
            rq2_alpha = file_cfg.get("alpha")
            if file_cfg.get("honest_power") is not None:
                rq2_honest_power = file_cfg["honest_power"]
            if file_cfg.get("rounds") is not None:
                rq2_rounds = file_cfg["rounds"]
            # proxy: CLI --proxy 优先，否则取配置文件中的全局 proxy
            if rq2_proxy is None and file_cfg.get("proxy") is not None:
                rq2_proxy = file_cfg["proxy"]
            print(f"  📄 已从 {args.config} 加载代理配置")
            if rq2_proxy:
                print(f"  🌐 代理设置: {rq2_proxy}")

        # ---- 方式 2: 从 --agents 命令行参数解析 ----
        elif args.agents:
            rq2_agent_configs = []
            for i, spec in enumerate(args.agents.split(",")):
                parts = spec.strip().split(":")
                if len(parts) == 2:
                    rq2_agent_configs.append({
                        "provider": parts[0],
                        "model": parts[1],
                        "name": f"矿工-{chr(65 + i)}-{parts[1][:10]}",
                    })
                else:
                    rq2_agent_configs.append({
                        "provider": args.provider,
                        "model": parts[0],
                        "name": f"矿工-{chr(65 + i)}",
                    })

        # 解析算力配置（命令行 --alpha 优先于配置文件）
        if args.alpha:
            rq2_alpha = [float(x.strip()) for x in args.alpha.split(",")]

        num_agents = len(rq2_agent_configs) if rq2_agent_configs else args.num_agents

        run_rq2(
            provider=args.provider, model=args.model,
            api_key=args.api_key, base_url=args.base_url,
            num_agents=num_agents, num_rounds=rq2_rounds,
            seed=args.seed,
            save_path=args.save or "reports/rq2_result.json",
            agent_configs=rq2_agent_configs,
            alpha=rq2_alpha,
            honest_power=rq2_honest_power,
            proxy=rq2_proxy,
        )

    elif args.mode == "rq3":
        run_rq3(
            provider=args.provider, model=args.model,
            api_key=args.api_key, base_url=args.base_url,
            num_rounds=args.rounds, seed=args.seed,
            save_path=args.save or "reports/rq3_result.json",
        )

    elif args.mode == "interactive":
        run_interactive(max_rounds=args.rounds, seed=args.seed)


if __name__ == "__main__":
    main()
