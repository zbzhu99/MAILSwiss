from marlkit.env_creators.highway.highway_env import HighwayEnv


def test_highway_env():
    env = HighwayEnv(
        env_name="merge-v2", env_kwargs={}
    )
    obs_n = env.reset()
    print(env.observation_space_n)
    print(obs_n)


if __name__ == "__main__":
    test_highway_env()
