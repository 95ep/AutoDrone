# Airsim gym environment # 

Install package with 

    pip install -e *path*/gym-airsim

There might be a need to the directory to PATH

# Example Use #
    import gym
    import gym_airsim
    env = gym.make('Airsim-v0', sensors=['rgb', 'depth', 'pointgoal_with_gps_compass'], max_dist=10)

After you have installed your package with pip install -e gym-foo, you can create an instance of the environment with gym.make('gym_foo:foo-v0')
