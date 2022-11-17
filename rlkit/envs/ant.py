import numpy as np

from gymnasium.envs.mujoco import AntEnv as AntEnv_
from gymnasium import spaces


class AntEnv(AntEnv_):
    def __init__(self):
        super().__init__()
        
        # Update observation space to match observation space returned by _get_obs(),
        # which is overwritten in this class and subclasses
        observation, _, _, _, _ = self.step(self.action_space.sample())
        obs_dim = observation.size
        high = np.inf*np.ones(obs_dim)
        low = -high
        self.observation_space = spaces.Box(low, high)
    
    def _get_obs(self):
        # this is gym ant obs, should use rllab?
        # if position is needed, override this in subclasses
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.standard_normal(size=self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
