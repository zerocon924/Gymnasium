"""Blockchain CPD (Constructive-Parasitic-Destructive) consensus game environment.

This environment models the resource allocation game among miners in a
blockchain network. Each miner allocates their computational resources
across three strategies:
    - Constructive (c): Honest mining that contributes to network security
    - Parasitic (p): Selfish mining that free-rides on others' honest work
    - Destructive (d): Attacking behaviour that undermines opponents' efficiency

The action space is a 3-dimensional continuous simplex (c + p + d = 1),
capturing the CPD resource allocation framework from the 2026 research design.

Key design insight:
    Opponent efficiency (eta_j) serves as the coupling term in the utility
    function. Heavy destruction reduces eta_j, but since parasitic gains
    are positively correlated with eta_j, over-destruction backfires on
    the attacker's own parasitic revenue — creating a non-trivial
    strategic equilibrium.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import gymnasium as gym
from gymnasium import spaces


class BlockchainCPDEnv(gym.Env[np.ndarray, np.ndarray]):
    """A multi-agent blockchain consensus game with CPD action space.

    This environment simulates a round-based mining game where ``num_miners``
    miners compete for block rewards. Each miner distributes their hash
    power across constructive, parasitic, and destructive strategies.

    Observation Space:
        A vector of shape ``(obs_dim,)`` containing:
            [0] cumulative_reward   : agent's cumulative reward so far
            [1] opponent_efficiency : average efficiency of opponents (eta_j)
            [2] self_hash_share     : agent's share of total hash power (alpha_i)
            [3] round_progress      : current_round / max_rounds (normalized)
            [4] last_reward         : reward obtained in the previous round
            [5] efficiency_delta    : change in opponent efficiency from last round

    Action Space:
        A 3-dimensional Box [0, 1]^3. The environment internally applies
        simplex normalization to ensure c + p + d = 1.

    Reward:
        Non-linear coupled utility:
            U_i = R * alpha_i * c_i                          (constructive)
                + R * p_i * (eta_j ^ beta)                   (parasitic)
                - lambda_ * (d_i ^ 2)                        (destruction cost)

    Args:
        num_miners: Number of miners in the game (default: 2).
        max_rounds: Maximum number of rounds per episode (default: 100).
        base_reward: Base block reward R per round (default: 10.0).
        alpha: Hash power shares for each miner. If None, equally distributed.
        beta: Convexity parameter for parasitic returns (default: 1.5).
        lambda_: Cost coefficient for destructive actions (default: 2.0).
        kappa: Impact factor of destruction on opponent efficiency (default: 0.3).
        eta_min: Minimum opponent efficiency floor (default: 0.1).
        eta_recovery: Per-round natural recovery rate of efficiency (default: 0.05).
        agent_id: Which miner this environment instance controls (default: 0).
        opponent_policy: Policy for non-controlled miners. One of
            'honest', 'random', 'tit_for_tat' (default: 'honest').
        render_mode: Rendering mode (default: None).
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        num_miners: int = 2,
        max_rounds: int = 100,
        base_reward: float = 10.0,
        alpha: np.ndarray | list[float] | None = None,
        beta: float = 1.5,
        lambda_: float = 2.0,
        kappa: float = 0.3,
        eta_min: float = 0.1,
        eta_recovery: float = 0.05,
        agent_id: int = 0,
        opponent_policy: str = "honest",
        render_mode: str | None = None,
    ):
        super().__init__()

        assert num_miners >= 2, "Need at least 2 miners for a game"
        assert 0 <= agent_id < num_miners, "agent_id must be in [0, num_miners)"
        assert opponent_policy in ("honest", "random", "tit_for_tat"), (
            f"Unknown opponent policy: {opponent_policy}"
        )

        self.num_miners = num_miners
        self.max_rounds = max_rounds
        self.base_reward = base_reward
        self.beta = beta
        self.lambda_ = lambda_
        self.kappa = kappa
        self.eta_min = eta_min
        self.eta_recovery = eta_recovery
        self.agent_id = agent_id
        self.opponent_policy = opponent_policy
        self.render_mode = render_mode

        # Hash power distribution (alpha_i for each miner)
        if alpha is not None:
            self.alpha = np.array(alpha, dtype=np.float64)
            assert len(self.alpha) == num_miners
            self.alpha = self.alpha / self.alpha.sum()  # normalize
        else:
            self.alpha = np.ones(num_miners, dtype=np.float64) / num_miners

        # --- Spaces ---
        # Action: 3D continuous vector [c, p, d], will be simplex-normalized
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float64
        )

        # Observation: 6-dimensional state vector
        self.obs_dim = 6
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, 0.0, 0.0, 0.0, -np.inf, -np.inf]),
            high=np.array([np.inf, 1.0, 1.0, 1.0, np.inf, np.inf]),
            shape=(self.obs_dim,),
            dtype=np.float64,
        )

        # --- Internal state (initialized in reset) ---
        self._cumulative_rewards: np.ndarray | None = None
        self._efficiencies: np.ndarray | None = None
        self._current_round: int = 0
        self._last_reward: float = 0.0
        self._prev_opponent_eta: float = 1.0
        self._last_actions: np.ndarray | None = None  # track for tit-for-tat
        self._history: list[dict] = []  # full round history for logging

    # ------------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to the initial state.

        Args:
            seed: Random seed for reproducibility.
            options: Optional dict. Supports:
                - 'alpha': override hash power distribution for this episode.

        Returns:
            observation: Initial observation for the controlled agent.
            info: Auxiliary information dict.
        """
        super().reset(seed=seed)

        # Allow overriding alpha via options
        if options and "alpha" in options:
            self.alpha = np.array(options["alpha"], dtype=np.float64)
            self.alpha = self.alpha / self.alpha.sum()

        # Reset internal state
        self._cumulative_rewards = np.zeros(self.num_miners, dtype=np.float64)
        self._efficiencies = np.ones(self.num_miners, dtype=np.float64)
        self._current_round = 0
        self._last_reward = 0.0
        self._prev_opponent_eta = 1.0
        self._last_actions = np.zeros((self.num_miners, 3), dtype=np.float64)
        self._last_actions[:, 0] = 1.0  # default: all honest
        self._history = []

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one round of the mining game.

        Args:
            action: A 3D vector [c, p, d] for the controlled agent.
                    Will be simplex-normalized internally.

        Returns:
            observation: New observation after the round.
            reward: The agent's utility for this round.
            terminated: Whether the episode has ended (max rounds reached).
            truncated: Always False (no external truncation).
            info: Auxiliary information including all miners' actions and rewards.
        """
        assert self._cumulative_rewards is not None, "Call reset() before step()"

        # 1. Normalize the agent's action to simplex
        agent_action = self._simplex_normalize(np.array(action, dtype=np.float64))

        # 2. Generate opponent actions
        all_actions = self._generate_all_actions(agent_action)

        # 3. Compute utilities for all miners
        rewards = self._compute_utilities(all_actions)

        # 4. Update efficiencies based on destructive actions
        prev_efficiencies = self._efficiencies.copy()
        self._update_efficiencies(all_actions)

        # 5. Update internal state
        self._cumulative_rewards += rewards
        self._current_round += 1
        agent_reward = float(rewards[self.agent_id])
        self._last_reward = agent_reward
        opponent_mask = np.ones(self.num_miners, dtype=bool)
        opponent_mask[self.agent_id] = False
        self._prev_opponent_eta = float(prev_efficiencies[opponent_mask].mean())
        self._last_actions = all_actions.copy()

        # 6. Record history
        round_record = {
            "round": self._current_round,
            "actions": all_actions.copy(),
            "rewards": rewards.copy(),
            "efficiencies": self._efficiencies.copy(),
            "cumulative_rewards": self._cumulative_rewards.copy(),
        }
        self._history.append(round_record)

        # 7. Check termination
        terminated = self._current_round >= self.max_rounds
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        info["round_record"] = round_record

        return obs, agent_reward, terminated, truncated, info

    def render(self) -> str | None:
        """Render the current game state as text."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    # ------------------------------------------------------------------
    # Utility computation
    # ------------------------------------------------------------------

    def _compute_utilities(self, all_actions: np.ndarray) -> np.ndarray:
        """Compute the non-linear coupled utility for each miner.

        Utility function for miner i:
            U_i = R * alpha_i * c_i                    (constructive revenue)
                + R * p_i * (mean_eta_opponents ^ beta) (parasitic revenue)
                - lambda_ * (d_i ^ 2)                  (destruction cost)

        The key coupling: parasitic gains depend on opponent efficiency,
        which is degraded by destructive actions from all miners.

        Args:
            all_actions: Array of shape (num_miners, 3) with [c, p, d] per miner.

        Returns:
            Array of shape (num_miners,) with utility for each miner.
        """
        rewards = np.zeros(self.num_miners, dtype=np.float64)

        for i in range(self.num_miners):
            c_i, p_i, d_i = all_actions[i]

            # Mean opponent efficiency
            opponent_mask = np.ones(self.num_miners, dtype=bool)
            opponent_mask[i] = False
            mean_eta_j = self._efficiencies[opponent_mask].mean()

            # Constructive revenue: honest mining proportional to hash share
            constructive = self.base_reward * self.alpha[i] * c_i

            # Parasitic revenue: depends on opponent efficiency (coupling!)
            parasitic = self.base_reward * p_i * (mean_eta_j ** self.beta)

            # Destruction cost: quadratic penalty (makes marginal cost increase)
            destruction_cost = self.lambda_ * (d_i ** 2)

            rewards[i] = constructive + parasitic - destruction_cost

        return rewards

    def _update_efficiencies(self, all_actions: np.ndarray) -> None:
        """Update mining efficiencies based on destructive actions.

        Each miner j's efficiency is reduced by the total destructive
        actions targeting them (from all other miners), then naturally
        recovers by eta_recovery per round.

        Args:
            all_actions: Array of shape (num_miners, 3) with [c, p, d] per miner.
        """
        for j in range(self.num_miners):
            # Total destruction aimed at miner j (from all others)
            opponent_mask = np.ones(self.num_miners, dtype=bool)
            opponent_mask[j] = False
            total_destruction = all_actions[opponent_mask, 2].sum()

            # Efficiency degradation
            self._efficiencies[j] -= self.kappa * total_destruction

            # Natural recovery
            self._efficiencies[j] += self.eta_recovery

            # Clamp to [eta_min, 1.0]
            self._efficiencies[j] = np.clip(
                self._efficiencies[j], self.eta_min, 1.0
            )

    # ------------------------------------------------------------------
    # Opponent policies
    # ------------------------------------------------------------------

    def _generate_all_actions(self, agent_action: np.ndarray) -> np.ndarray:
        """Generate actions for all miners including opponents.

        Args:
            agent_action: Simplex-normalized action for the controlled agent.

        Returns:
            Array of shape (num_miners, 3) with actions for all miners.
        """
        all_actions = np.zeros((self.num_miners, 3), dtype=np.float64)
        all_actions[self.agent_id] = agent_action

        for i in range(self.num_miners):
            if i == self.agent_id:
                continue
            all_actions[i] = self._get_opponent_action(i)

        return all_actions

    def _get_opponent_action(self, miner_id: int) -> np.ndarray:
        """Get action for an opponent miner based on the opponent policy.

        Args:
            miner_id: The opponent miner's index.

        Returns:
            Simplex-normalized action [c, p, d].
        """
        if self.opponent_policy == "honest":
            # Pure honest mining: all resources to constructive
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        elif self.opponent_policy == "random":
            # Random Dirichlet allocation on the simplex
            raw = self.np_random.dirichlet(np.ones(3))
            return raw.astype(np.float64)

        elif self.opponent_policy == "tit_for_tat":
            # Mirror the agent's previous action
            if self._last_actions is not None:
                return self._last_actions[self.agent_id].copy()
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # Fallback
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)

    # ------------------------------------------------------------------
    # Observation & info helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector for the controlled agent.

        Returns:
            obs: Array of shape (obs_dim,) = [
                cumulative_reward,
                mean_opponent_efficiency,
                self_hash_share,
                round_progress,
                last_reward,
                efficiency_delta
            ]
        """
        cum_reward = (
            self._cumulative_rewards[self.agent_id]
            if self._cumulative_rewards is not None
            else 0.0
        )

        # Opponent mean efficiency
        if self._efficiencies is not None:
            opp_mask = np.ones(self.num_miners, dtype=bool)
            opp_mask[self.agent_id] = False
            mean_opp_eta = float(self._efficiencies[opp_mask].mean())
        else:
            mean_opp_eta = 1.0

        alpha_i = float(self.alpha[self.agent_id])
        round_progress = self._current_round / max(self.max_rounds, 1)
        efficiency_delta = mean_opp_eta - self._prev_opponent_eta

        return np.array(
            [
                cum_reward,
                mean_opp_eta,
                alpha_i,
                round_progress,
                self._last_reward,
                efficiency_delta,
            ],
            dtype=np.float64,
        )

    def _get_info(self) -> dict[str, Any]:
        """Build the info dict with diagnostic data.

        Returns:
            info dict containing efficiencies, cumulative rewards, etc.
        """
        return {
            "current_round": self._current_round,
            "efficiencies": (
                self._efficiencies.copy()
                if self._efficiencies is not None
                else None
            ),
            "cumulative_rewards": (
                self._cumulative_rewards.copy()
                if self._cumulative_rewards is not None
                else None
            ),
            "alpha": self.alpha.copy(),
            "history_length": len(self._history),
        }

    # ------------------------------------------------------------------
    # Simplex normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _simplex_normalize(action: np.ndarray) -> np.ndarray:
        """Project an action vector onto the probability simplex.

        Ensures all components are non-negative and sum to 1.
        If all components are zero or negative, defaults to pure honest mining.

        Args:
            action: Raw 3D action vector [c, p, d].

        Returns:
            Normalized action on the simplex.
        """
        # Clamp negative values to 0
        action = np.maximum(action, 0.0)
        total = action.sum()

        if total < 1e-8:
            # Fallback: pure constructive (honest mining)
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)

        return action / total

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_ansi(self) -> str:
        """Render game state as ANSI text."""
        lines = [
            f"=== Blockchain CPD Game | Round {self._current_round}/{self.max_rounds} ===",
        ]

        if self._efficiencies is not None and self._cumulative_rewards is not None:
            for i in range(self.num_miners):
                role = "Agent" if i == self.agent_id else "Opponent"
                last_act = (
                    self._last_actions[i]
                    if self._last_actions is not None
                    else [0, 0, 0]
                )
                lines.append(
                    f"  Miner {i} ({role}): "
                    f"alpha={self.alpha[i]:.2f}, "
                    f"eta={self._efficiencies[i]:.3f}, "
                    f"cum_R={self._cumulative_rewards[i]:.2f}, "
                    f"last_action=[c={last_act[0]:.2f}, p={last_act[1]:.2f}, d={last_act[2]:.2f}]"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # History access (for memory module)
    # ------------------------------------------------------------------

    def get_history(self) -> list[dict]:
        """Return the complete game history for analysis and memory modules.

        Returns:
            List of round records, each containing actions, rewards,
            efficiencies, and cumulative rewards.
        """
        return self._history.copy()

    def get_last_n_rounds(self, n: int) -> list[dict]:
        """Return the last n rounds of history.

        Args:
            n: Number of recent rounds to retrieve.

        Returns:
            List of the most recent round records.
        """
        return self._history[-n:] if self._history else []


# ======================================================================
# 多智能体版本（POMG）：LLM 代理 + 背景诚实算力群体
# ======================================================================


class MultiAgentBlockchainCPDEnv:
    """多智能体区块链 CPD 博弈环境（POMG 建模）。

    核心设计（对照 plan.md §1-§2）：
    ─────────────────────────────────────────────────────────────
    建模为 **部分观测马尔可夫博弈 (Partially Observed Markov Game)**：
    - N 个 LLM 代理节点 + 1 个 **背景诚实算力群体 (Honest Group)**
    - 总算力 = 1.0。代理算力之和 < 1，剩余部分属于诚实群体
    - 诚实群体始终执行纯建设策略 [1, 0, 0]，代表协议底线
    - 代理的破坏/寄生行为对诚实群体同样生效（动态耦合）

    算力设定（非饱和分配规则）：
        例: alpha=[0.25, 0.20, 0.15]  → 代理总算力=0.60
            honest_power=0.40          → 诚实群体
            全网总算力 = 1.00
        代理不仅与彼此博弈，还与诚实群体竞争，
        这使得"共同剥削诚实算力"的寄生均衡成为可能。

    状态互感（Inter-observability）：
        每个代理的 info 包含：
        - 其他 LLM 代理的公开状态（累计收益、效率η、策略倾向标签）
        - 背景诚实群体的效率状态

    动态耦合效应：
        代理 i 的破坏 d_i 同时影响所有其他节点（包括诚实群体）的效率η。

    Utility 函数：
        U_i = R · α_i · c_i                           (建设收益)
            + R · p_i · (mean_η_opponents ^ β)         (寄生收益)
            - λ · d_i²                                  (破坏成本)

    Args:
        num_agents: LLM 代理数量（默认 3）。
        max_rounds: 每局最大轮次（默认 100）。
        base_reward: 基础区块奖励 R（默认 10.0）。
        alpha: LLM 代理的算力列表。sum(alpha) < 1.0。
        honest_power: 背景诚实群体的算力（默认 0.40）。
        beta: 寄生收益凸性参数（默认 1.5）。
        lambda_: 破坏成本系数（默认 2.0）。
        kappa: 破坏对效率的影响因子（默认 0.3）。
        eta_min: 效率下界（默认 0.1）。
        eta_recovery: 每轮自然恢复量（默认 0.05）。
    """

    # 诚实群体的内部索引（始终为最后一个）
    HONEST_GROUP_ACTION = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    def __init__(
        self,
        num_agents: int = 3,
        max_rounds: int = 100,
        base_reward: float = 10.0,
        alpha: list[float] | np.ndarray | None = None,
        honest_power: float = 0.40,
        beta: float = 1.5,
        lambda_: float = 2.0,
        kappa: float = 0.3,
        eta_min: float = 0.1,
        eta_recovery: float = 0.05,
    ):
        assert num_agents >= 2, "至少需要 2 个 LLM 代理"

        self.num_agents = num_agents
        # 总参与者 = LLM 代理 + 诚实群体
        self.num_miners = num_agents + 1
        self.honest_id = num_agents  # 诚实群体在内部数组中的索引
        self.honest_power = honest_power
        self.max_rounds = max_rounds
        self.base_reward = base_reward
        self.beta = beta
        self.lambda_ = lambda_
        self.kappa = kappa
        self.eta_min = eta_min
        self.eta_recovery = eta_recovery

        # ---- 算力分配 ----
        if alpha is not None:
            agent_alpha = np.array(alpha, dtype=np.float64)
            assert len(agent_alpha) == num_agents, (
                f"alpha 长度 ({len(agent_alpha)}) 必须等于 num_agents ({num_agents})"
            )
        else:
            agent_alpha = np.array(
                [0.25, 0.20, 0.15][:num_agents], dtype=np.float64
            )

        # 完整算力数组 = [代理_0, ..., 代理_{N-1}, 诚实群体]
        self._full_alpha = np.append(agent_alpha, honest_power)
        total_hp = float(self._full_alpha.sum())
        assert abs(total_hp - 1.0) < 0.05, (
            f"总算力应接近 1.0, 实际为 {total_hp:.4f} "
            f"(agents={agent_alpha.tolist()}, honest={honest_power})"
        )

        # 仅代理的算力视图（供外部访问）
        self.alpha = agent_alpha.copy()

        # ---- 内部状态 ----
        n = self.num_miners
        self._cumulative_rewards: np.ndarray = np.zeros(n, dtype=np.float64)
        self._efficiencies: np.ndarray = np.ones(n, dtype=np.float64)
        self._current_round: int = 0
        self._last_rewards: np.ndarray = np.zeros(n, dtype=np.float64)
        self._prev_efficiencies: np.ndarray = np.ones(n, dtype=np.float64)
        self._last_actions: np.ndarray = np.zeros((n, 3), dtype=np.float64)
        self._last_actions[:, 0] = 1.0  # 默认全诚实
        self._history: list[dict] = []

    # ------------------------------------------------------------------
    # 核心 API
    # ------------------------------------------------------------------

    def reset(
        self, seed: int | None = None
    ) -> tuple[dict[int, np.ndarray], dict[int, dict]]:
        """重置环境，返回各 LLM 代理的初始观测。"""
        n = self.num_miners
        self._cumulative_rewards = np.zeros(n, dtype=np.float64)
        self._efficiencies = np.ones(n, dtype=np.float64)
        self._current_round = 0
        self._last_rewards = np.zeros(n, dtype=np.float64)
        self._prev_efficiencies = np.ones(n, dtype=np.float64)
        self._last_actions = np.zeros((n, 3), dtype=np.float64)
        self._last_actions[:, 0] = 1.0
        self._history = []

        obs_dict: dict[int, np.ndarray] = {}
        info_dict: dict[int, dict] = {}
        for i in range(self.num_agents):
            obs_dict[i] = self._get_obs(i)
            info_dict[i] = self._get_info(i)
        return obs_dict, info_dict

    def step(
        self, actions: dict[int, np.ndarray]
    ) -> tuple[dict[int, np.ndarray], dict[int, float], bool, dict[int, dict]]:
        """执行一轮博弈。

        Args:
            actions: 仅 LLM 代理的动作 {0: act_0, 1: act_1, ...}。
                     诚实群体的动作由环境自动生成。

        Returns:
            obs_dict, rewards_dict, terminated, info_dict
            （仅包含 LLM 代理的数据，不含诚实群体）
        """
        # 1. 构建全部动作矩阵（代理动作 + 诚实群体固定动作）
        all_actions = np.zeros((self.num_miners, 3), dtype=np.float64)
        for i in range(self.num_agents):
            raw = np.array(actions[i], dtype=np.float64)
            all_actions[i] = self._simplex_normalize(raw)
        all_actions[self.honest_id] = self.HONEST_GROUP_ACTION.copy()

        # 2. 计算 Utility（包含诚实群体）
        rewards = self._compute_utilities(all_actions)

        # 3. 更新效率（诚实群体也会被攻击，也会恢复）
        self._prev_efficiencies = self._efficiencies.copy()
        self._update_efficiencies(all_actions)

        # 4. 更新状态
        self._cumulative_rewards += rewards
        self._current_round += 1
        self._last_rewards = rewards.copy()
        self._last_actions = all_actions.copy()

        # 5. 历史记录
        round_record = {
            "round": self._current_round,
            "actions": all_actions.copy(),
            "rewards": rewards.copy(),
            "efficiencies": self._efficiencies.copy(),
            "cumulative_rewards": self._cumulative_rewards.copy(),
        }
        self._history.append(round_record)

        # 6. 终止判断
        terminated = self._current_round >= self.max_rounds

        # 7. 仅为 LLM 代理构建返回值
        obs_dict: dict[int, np.ndarray] = {}
        rewards_dict: dict[int, float] = {}
        info_dict: dict[int, dict] = {}
        for i in range(self.num_agents):
            obs_dict[i] = self._get_obs(i)
            rewards_dict[i] = float(rewards[i])
            info_dict[i] = self._get_info(i)
            info_dict[i]["round_record"] = round_record

        return obs_dict, rewards_dict, terminated, info_dict

    # ------------------------------------------------------------------
    # Utility / 效率
    # ------------------------------------------------------------------

    def _compute_utilities(self, all_actions: np.ndarray) -> np.ndarray:
        """计算所有参与者（含诚实群体）的效用。"""
        rewards = np.zeros(self.num_miners, dtype=np.float64)
        for i in range(self.num_miners):
            c_i, p_i, d_i = all_actions[i]
            opp_mask = np.ones(self.num_miners, dtype=bool)
            opp_mask[i] = False
            mean_eta_j = self._efficiencies[opp_mask].mean()

            constructive = self.base_reward * self._full_alpha[i] * c_i
            parasitic = self.base_reward * p_i * (mean_eta_j ** self.beta)
            destruction_cost = self.lambda_ * (d_i ** 2)
            rewards[i] = constructive + parasitic - destruction_cost
        return rewards

    def _update_efficiencies(self, all_actions: np.ndarray) -> None:
        """更新效率。诚实群体也可被攻击、也能恢复。"""
        for j in range(self.num_miners):
            opp_mask = np.ones(self.num_miners, dtype=bool)
            opp_mask[j] = False
            total_d = all_actions[opp_mask, 2].sum()
            self._efficiencies[j] -= self.kappa * total_d
            self._efficiencies[j] += self.eta_recovery
            self._efficiencies[j] = np.clip(
                self._efficiencies[j], self.eta_min, 1.0
            )

    # ------------------------------------------------------------------
    # 观测 / 信息
    # ------------------------------------------------------------------

    def _get_obs(self, agent_id: int) -> np.ndarray:
        """为 LLM 代理构建 6 维观测。

        对手效率 = 其他所有参与者（含诚实群体）的 η 均值。
        """
        cum_r = float(self._cumulative_rewards[agent_id])

        opp_mask = np.ones(self.num_miners, dtype=bool)
        opp_mask[agent_id] = False
        mean_opp_eta = float(self._efficiencies[opp_mask].mean())

        alpha_i = float(self._full_alpha[agent_id])
        progress = self._current_round / max(self.max_rounds, 1)
        last_r = float(self._last_rewards[agent_id])

        prev_opp_eta = float(self._prev_efficiencies[opp_mask].mean())
        eff_delta = mean_opp_eta - prev_opp_eta

        return np.array(
            [cum_r, mean_opp_eta, alpha_i, progress, last_r, eff_delta],
            dtype=np.float64,
        )

    def _get_info(self, agent_id: int) -> dict[str, Any]:
        """为 LLM 代理构建信息字典（含社会感知信息）。

        包含：
        - other_agents: 其他 LLM 代理的公开状态 + 策略倾向标签
        - honest_group: 背景诚实群体的状态
        """
        # ---- 其他 LLM 代理 ----
        other_agents_info = []
        for j in range(self.num_agents):
            if j == agent_id:
                continue
            act = self._last_actions[j]
            other_agents_info.append({
                "agent_id": j,
                "last_action": act.tolist(),
                "efficiency": float(self._efficiencies[j]),
                "cumulative_reward": float(self._cumulative_rewards[j]),
                "hash_power": float(self._full_alpha[j]),
                "strategy_label": self._derive_strategy_label(act),
            })

        # ---- 背景诚实群体 ----
        h = self.honest_id
        honest_info = {
            "hash_power": float(self._full_alpha[h]),
            "efficiency": float(self._efficiencies[h]),
            "cumulative_reward": float(self._cumulative_rewards[h]),
            "action": self._last_actions[h].tolist(),
        }

        return {
            "current_round": self._current_round,
            "efficiencies": self._efficiencies.copy(),
            "cumulative_rewards": self._cumulative_rewards.copy(),
            "alpha": self._full_alpha.copy(),
            "agent_id": agent_id,
            "num_agents": self.num_agents,
            "other_agents": other_agents_info,
            "honest_group": honest_info,
            "history_length": len(self._history),
        }

    @staticmethod
    def _derive_strategy_label(action: np.ndarray) -> str:
        """根据动作向量推导策略倾向标签。"""
        c, p, d = float(action[0]), float(action[1]), float(action[2])
        if c >= 0.6:
            return "诚实建设者"
        if d >= 0.25:
            return "攻击者"
        if p >= 0.5:
            return "寄生搭便车者"
        if p >= 0.3 and d >= 0.15:
            return "机会主义者"
        return "均衡策略者"

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @staticmethod
    def _simplex_normalize(action: np.ndarray) -> np.ndarray:
        action = np.maximum(action, 0.0)
        total = action.sum()
        if total < 1e-8:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return action / total

    def get_history(self) -> list[dict]:
        return self._history.copy()

    def get_last_n_rounds(self, n: int) -> list[dict]:
        return self._history[-n:] if self._history else []

    def render(self) -> str:
        lines = [
            f"=== 多智能体 POMG | 轮 {self._current_round}/{self.max_rounds} ===",
            f"    算力: agents={self.alpha.tolist()}, "
            f"honest={self.honest_power}, total={self._full_alpha.sum():.2f}",
        ]
        for i in range(self.num_agents):
            act = self._last_actions[i]
            label = self._derive_strategy_label(act)
            lines.append(
                f"  代理{i}: α={self._full_alpha[i]:.2f}, "
                f"η={self._efficiencies[i]:.3f}, "
                f"R={self._cumulative_rewards[i]:.2f}, "
                f"[c={act[0]:.2f},p={act[1]:.2f},d={act[2]:.2f}] ({label})"
            )
        h = self.honest_id
        lines.append(
            f"  诚实群体: α={self._full_alpha[h]:.2f}, "
            f"η={self._efficiencies[h]:.3f}, "
            f"R={self._cumulative_rewards[h]:.2f} (纯建设)"
        )
        return "\n".join(lines)

    def close(self) -> None:
        pass
