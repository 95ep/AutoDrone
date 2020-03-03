# Airgym environment # 

Gym-like API, which can be used to create a reinforcement learning environment on top of airsim. 

# Example Use #
    import airgym
    env = airgym.make(sensors=['rgb', 'depth', 'pointgoal_with_gps_compass'], max_dist=10)
    obs = env.reset()
    obs, reward, done, (position, orientation)
