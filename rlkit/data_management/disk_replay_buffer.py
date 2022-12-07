import tempfile, os
import numpy as np

from .simple_replay_buffer import SimpleReplayBuffer

class DiskReplayBuffer(SimpleReplayBuffer):
    def __init__(
            self, max_replay_buffer_size, observation_dim, action_dim
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        
        self._storage_dir = tempfile.TemporaryDirectory()
        self._storage = np.memmap(os.path.join(self._storage_dir.name, "replay_buffer.tmp"), dtype=np.float32, mode="w+", shape=(max_replay_buffer_size, 2 * observation_dim + action_dim + 3))
        
        self.clear()
        
    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        combined = np.concatenate([
            observation,
            action,
            np.atleast_1d(reward),
            np.atleast_1d(terminal),
            next_observation,
            np.atleast_1d(kwargs['env_info'].get('sparse_reward', 0))
        ], axis=0)
        self._storage[self._top] = combined
        self._advance()

    def sample_data(self, indices):
        combined = self._storage[indices]
        observations, actions, rewards, terminals, next_observations, sparse_rewards = np.split(
            combined,
            [
                self._observation_dim,
                self._observation_dim + self._action_dim,
                self._observation_dim + self._action_dim + 1,
                self._observation_dim + self._action_dim + 2,
                self._observation_dim + self._action_dim + 2 + self._observation_dim
            ],
            axis=1
        )
        return dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals.astype(np.uint8),
            next_observations=next_observations,
            sparse_rewards=sparse_rewards
        )
        
    def __del__(self):
        del self._storage
        del self._storage_dir