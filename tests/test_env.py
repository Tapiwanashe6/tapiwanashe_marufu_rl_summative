import numpy as np
from gymnasium.utils.env_checker import check_env
from environment import MediFlightEnv


def test_gymnasium_contract():
    check_env(MediFlightEnv(), skip_render_check=True)


def test_seed_reproducibility():
    a, b = MediFlightEnv(), MediFlightEnv()
    np.testing.assert_array_equal(a.reset(seed=12)[0], b.reset(seed=12)[0])
    for action in [0, 1, 13, 2, 0, 4]:
        x, y = a.step(action), b.step(action)
        np.testing.assert_array_equal(x[0], y[0])
        assert x[1:] == y[1:]


def test_episode_finishes():
    env = MediFlightEnv()
    env.reset(seed=2)
    for _ in range(100):
        _, _, term, trunc, _ = env.step(0)
        if term or trunc:
            break
    assert trunc
