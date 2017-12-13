from gym.envs.registration import register

register(
    id='GridWorldEnv-v0',
    entry_point='Gym_GridWorld.envs:GridWorldEnv',
)