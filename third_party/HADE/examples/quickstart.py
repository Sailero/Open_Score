"""Run with `python examples/quickstart.py` after `pip install -e .`."""
from had_env import make_env


def main():
    env = make_env("defense", red_count=4, blue_count=4, max_cycles=50)
    observations, infos = env.reset(seed=42)
    try:
        while env.agents:
            actions = {name: env.action_space(name).sample() for name in env.agents}
            observations, rewards, terminated, truncated, infos = env.step(actions)
        print(f"steps={env.num_cycles}, native_red_outcome={env.outcome_red}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
