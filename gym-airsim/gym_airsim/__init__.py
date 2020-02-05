from gym.envs.registration import register

register(
    id='Airsim-v0',
    entry_point='gym_airsim.envs:AirsimEnv',
)


"""    ÖVRIGA PARAMETRAR EXEMPELVIS:
timestep_limit=1000,
reward_threshold=1.0,
kwargs (dict): The kwargs to pass to the environment class
"""


"""
register(
    id='AirsimAlternativeEnvironment-v0',
    entry_point='gym_foo.envs:FooExtraHardEnv',
    reward_threshold=10.0,
)
"""