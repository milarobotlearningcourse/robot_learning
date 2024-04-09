from gym.envs.registration import register

def register_envs():
    register(
        id='cheetah-roble-v0',
        entry_point='hw2.roble.envs.cheetah:HalfCheetahEnv',
        max_episode_steps=1000,
    )
    register(
        id='obstacles-roble-v0',
        entry_point='hw2.roble.envs.obstacles:Obstacles',
        max_episode_steps=500,
    )
    register(
        id='reacher-roble-v0',
        entry_point='hw2.roble.envs.reacher:Reacher7DOFEnv',
        max_episode_steps=500,
    )
