import numpy as np
from finrl_goexplore.make_env import make_finrl_env


def main():
    env_id = "finrl:finrl_goexplore/data/aapl_2016_2020.csv:AAPL"
    env = make_finrl_env(env_id, seed=123)

    obs, _ = env.reset()
    acts = [0, 8, 4, 8, 0, 4, 8]

    for a in acts[:3]:
        obs, r, done, trunc, info = env.step(a)

    snap = env.clone_state()

    seq1 = []
    for a in acts[3:]:
        obs, r, done, trunc, info = env.step(a)
        seq1.append((obs.sum(), r, done))

    env.restore_state(snap)

    seq2 = []
    for a in acts[3:]:
        obs, r, done, trunc, info = env.step(a)
        seq2.append((obs.sum(), r, done))

    ok = np.allclose(np.array(seq1, dtype=float), np.array(seq2, dtype=float))
    print("snapshot deterministic:", ok)


if __name__ == "__main__":
    main()
