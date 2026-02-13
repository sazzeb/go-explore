from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional
import copy

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.config import INDICATORS


@dataclass
class FinRLGoExploreSpec:
    csv_path: str
    tickers: Tuple[str, ...] = ("AAPL",)
    initial_amount: int = 1_000_000
    hmax: int = 100
    buy_cost_pct: float = 0.001
    sell_cost_pct: float = 0.001
    reward_scaling: float = 1e-4
    frame_h: int = 84
    frame_w: int = 84
    action_bins: int = 9


@dataclass(frozen=True)
class FinRLPos:
    # Minimal "position" object so Go-Explore can discretize and log progress.
    score: int = 0
    day: int = 0
    level: int = 0
    room: int = 0
    x: int = 0
    y: int = 0


def _load_df(csv_path: str, tickers: Tuple[str, ...]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["tic"].isin(list(tickers))].copy()
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    df = df.copy()
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df = df.reset_index(drop=True)

    df = df.set_index(df.groupby("date").ngroup())
    return df


def _state_to_frame(state: np.ndarray, h: int, w: int) -> np.ndarray:
    x = np.tanh(state / (np.std(state) + 1e-8))
    x = (x + 1.0) * 0.5
    x = np.clip(x, 0.0, 1.0)

    flat = np.zeros(h * w, dtype=np.float32)
    n = min(flat.size, x.size)
    flat[:n] = x[:n]

    frame = (flat.reshape(h, w) * 255.0).astype(np.uint8)
    frame = np.repeat(frame[..., None], 3, axis=2)
    return frame


class FinRLGoExploreEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, spec: FinRLGoExploreSpec, seed: int = 0):
        super().__init__()
        self.spec = spec
        self.seed_value = seed

        self.df = _load_df(spec.csv_path, spec.tickers)

        self.stock_dim = len(spec.tickers)
        self.tech_indicator_list = list(INDICATORS)

        self.state_space = 1 + 2 * self.stock_dim + len(self.tech_indicator_list) * self.stock_dim
        self.action_dim = self.stock_dim

        self._env = StockTradingEnv(
            df=self.df,
            stock_dim=self.stock_dim,
            hmax=spec.hmax,
            initial_amount=spec.initial_amount,
            num_stock_shares=[0] * self.stock_dim,
            buy_cost_pct=[spec.buy_cost_pct] * self.stock_dim,
            sell_cost_pct=[spec.sell_cost_pct] * self.stock_dim,
            reward_scaling=spec.reward_scaling,
            state_space=self.state_space,
            action_space=self.action_dim,
            tech_indicator_list=self.tech_indicator_list,
            turbulence_threshold=None,
            risk_indicator_col="turbulence",
        )

        self.action_space = Discrete(spec.action_bins)
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(spec.frame_h, spec.frame_w, 3),
            dtype=np.uint8,
        )

        self._last_state_vec: Optional[np.ndarray] = None

        # Go-Explore expects an env.state list of frames.
        self.state = []
        # Some Go-Explore code assumes this exists.
        self.rooms: Dict[Any, Any] = {}

    def _discrete_to_action(self, a: int) -> np.ndarray:
        bins = self.spec.action_bins
        v = -1.0 + 2.0 * (a / (bins - 1))
        return np.array([v] * self.stock_dim, dtype=np.float32)

    def reset(self):
        # Go-Explore uses the old Gym API: reset() -> observation
        reset_out = self._env.reset()
        obs_vec = reset_out[0] if isinstance(reset_out, tuple) and len(reset_out) == 2 else reset_out
        obs_vec = np.asarray(obs_vec, dtype=np.float32)
        self._last_state_vec = obs_vec
        frame = _state_to_frame(obs_vec, self.spec.frame_h, self.spec.frame_w)
        self.state = [frame]
        return copy.copy(self.state)

    def step(self, action: int):
        # Go-Explore uses the old Gym API: step() -> (obs, reward, done, info)
        act = self._discrete_to_action(int(action))
        step_out = self._env.step(act)
        if isinstance(step_out, tuple) and len(step_out) == 5:
            obs_vec, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
        else:
            obs_vec, reward, done, info = step_out
        obs_vec = np.asarray(obs_vec, dtype=np.float32)
        self._last_state_vec = obs_vec
        frame = _state_to_frame(obs_vec, self.spec.frame_h, self.spec.frame_w)
        self.state = [frame]
        return copy.copy(self.state), float(reward), bool(done), info

    def render(self):
        if self._last_state_vec is None:
            z = np.zeros((self.spec.frame_h, self.spec.frame_w, 3), dtype=np.uint8)
            return z
        return _state_to_frame(self._last_state_vec, self.spec.frame_h, self.spec.frame_w)

    def close(self):
        return

    def get_pos(self) -> FinRLPos:
        # Use day index as a simple monotonically increasing progress signal.
        day = int(getattr(self._env, "day", 0))
        return FinRLPos(score=day, day=day)

    def get_restore(self):
        # Match the Go-Explore env interface.
        return (
            self.clone_state(),
            copy.copy(self.state),
        )

    def restore(self, data):
        snap, state = data
        # Ensure internal structures exist, then restore.
        try:
            _ = self._env.reset()
        except Exception:
            pass
        self.restore_state(snap)
        self.state = copy.copy(state)

    def clone_state(self) -> Dict[str, Any]:
        e = self._env
        snap = {
            "day": e.day,
            "terminal": e.terminal,
            "state": np.array(e.state, dtype=np.float32),
            "reward": float(e.reward),
            "turbulence": float(getattr(e, "turbulence", 0.0)),
            "cost": float(getattr(e, "cost", 0.0)),
            "trades": int(getattr(e, "trades", 0)),
            "episode": int(getattr(e, "episode", 0)),
            "asset_memory": list(getattr(e, "asset_memory", [])),
            "rewards_memory": list(getattr(e, "rewards_memory", [])),
            "actions_memory": list(getattr(e, "actions_memory", [])),
            "state_memory": list(getattr(e, "state_memory", [])),
            "date_memory": list(getattr(e, "date_memory", [])),
        }

        rng_state = None
        if hasattr(e, "np_random") and hasattr(e.np_random, "bit_generator"):
            rng_state = e.np_random.bit_generator.state
        snap["rng_state"] = rng_state

        return snap

    def restore_state(self, snap: Dict[str, Any]) -> None:
        e = self._env
        e.day = int(snap["day"])
        e.data = e.df.loc[e.day, :]
        e.terminal = bool(snap["terminal"])
        e.state = np.array(snap["state"], dtype=np.float32).tolist()
        e.reward = float(snap["reward"])
        e.turbulence = float(snap["turbulence"])
        e.cost = float(snap["cost"])
        e.trades = int(snap["trades"])
        e.episode = int(snap["episode"])
        e.asset_memory = list(snap["asset_memory"])
        e.rewards_memory = list(snap["rewards_memory"])
        e.actions_memory = list(snap["actions_memory"])
        e.state_memory = list(snap["state_memory"])
        e.date_memory = list(snap["date_memory"])

        if snap.get("rng_state", None) is not None and hasattr(e, "np_random") and hasattr(e.np_random, "bit_generator"):
            e.np_random.bit_generator.state = snap["rng_state"]

        self._last_state_vec = np.asarray(e.state, dtype=np.float32)

    def clone_full_state(self):
        return self.clone_state()

    def restore_full_state(self, s):
        self.restore_state(s)
