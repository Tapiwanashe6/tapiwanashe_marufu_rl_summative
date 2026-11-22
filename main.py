"""Reproducible first-run demonstration (random policy baseline)."""
from environment import MediFlightEnv


def main():
    env = MediFlightEnv(render_mode="ansi")
    obs, info = env.reset(seed=42)
    print("MediFlight AI — random baseline")
    print(env.render())
    total = 0.0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        print(f"step={step:03d} action={action:02d} event={info['event']:<24} "
              f"reward={reward:7.1f} battery={info['battery']:5.1f} "
              f"time={info['time_min']:6.1f}")
        if terminated or truncated:
            break
    print(f"Episode reward={total:.1f} | {info}")
    env.close()


if __name__ == "__main__":
    main()
