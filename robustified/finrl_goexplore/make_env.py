# robustified/finrl_goexplore/make_env.py
from __future__ import annotations

import os
from finrl_goexplore.finrl_env import FinRLGoExploreEnv, FinRLGoExploreSpec


def make_finrl_env(env_id: str, seed: int = 0) -> FinRLGoExploreEnv:
    """
    env_id format:
      finrl:/path/to/data.csv:TICKER1,TICKER2,...
    Example:
      finrl:finrl_goexplore/data/aapl_2016_2020.csv:AAPL
    """
    parts = env_id.split(":")
    if len(parts) < 3 or parts[0] != "finrl":
        raise ValueError("env_id must look like finrl:/path/to.csv:TICKER1,TICKER2,...")

    csv_path = ":".join(parts[1:-1]).strip()
    tickers_str = parts[-1].strip()

    if not csv_path:
        raise ValueError("missing csv path in env_id")
    if not tickers_str:
        raise ValueError("missing tickers in env_id")

    if not os.path.isabs(csv_path):
        csv_path = os.path.abspath(csv_path)

    tickers = tuple(t.strip().upper() for t in tickers_str.split(",") if t.strip())
    if len(tickers) == 0:
        raise ValueError("no valid tickers parsed from env_id")

    spec = FinRLGoExploreSpec(
        csv_path=csv_path,
        tickers=tickers,
    )
    return FinRLGoExploreEnv(spec, seed=seed)


class FinRLGoExploreEnvFromId:
    """Pickle-friendly wrapper so Go-Explore can construct the env in subprocesses.

    Go-Explore expects an env class and kwargs; it will call `env_class(**kwargs)`.
    """

    TARGET_SHAPE = None
    MAX_PIX_VALUE = None

    def __init__(self, env_id: str, seed: int = 0):
        self._env = make_finrl_env(env_id, seed=seed)

    def __getattr__(self, name):
        return getattr(self._env, name)
